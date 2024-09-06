import os
import shutil
import h5py
import numpy as np
import pandas as pd
import hail as hl
from heig.wgs.staar import VariantSetTest, prepare_vset_test
import heig.input.dataset as ds
from heig.wgs.utils import *

"""
TODO: How to deal with multiple hypothesis testing issue?

"""

class SlidingWindow:
    def __init__(self, snps_mt, chr, start, end, variant_type, 
                 window_length, use_annotation_weights=True):
        """
        The specific type of variants have been extracted; genotype data has aligned with covariates
        
        """
        self.snps_mt = snps_mt
        self.variant_type = variant_type
        self.window_length = window_length
        self.start = start
        self.end = end
        self.windows = self._get_windows()

        chr = str(chr)
        if hl.default_reference == 'GRCh38':
            chr = 'chr' + chr
        self.chr = chr

        if variant_type == 'snv' and use_annotation_weights:
            self.annot_cols = [Annotation_name_catalog[annot_name] for annot_name in Annotation_name]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None

    def _get_windows(self):
        windows = list()
        sliding_length = self.window_length // 2
        cur_left = self.start
        cur_right = self.start + sliding_length
        while cur_right < self.end:
            windows.append(tuple([cur_left, cur_right]))
            cur_left, cur_right = cur_right, cur_right + sliding_length
        windows.append(tuple([cur_left, self.end]))

        return windows

    def extract_window(self, start_loc, end_loc):
        window_mt = self.snps_mt.filter_rows(
            (self.snps_mt.locus.contig == self.chr) & 
            (self.snps_mt.locus.position >= start_loc) & 
            (self.snps_mt.locus.position < end_loc)
        )

        return window_mt
        

def single_gene_analysis(snps_mt, chr, start, end, variant_type, window_length, 
                         use_annotation_weights, vset_test, log):
    """
    Single gene analysis

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated vcf
    start: start position of gene
    end: end position of gene
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    slidingwindow = SlidingWindow(snps_mt, chr, start, end, variant_type,
                                   window_length, use_annotation_weights)
    window_pvalues = dict()
    for start_loc, end_loc in slidingwindow.windows:
        snps_mt_cate = slidingwindow.extract_window(start_loc, end_loc)
        if snps_mt_cate.rows().count() <= 1:
            log.info(f'Less than 2 variants for window ({start_loc}, {end_loc}), skip.')
            continue
        if slidingwindow.annot_cols is not None:
            annot_phred = snps_mt_cate.fa.select(*slidingwindow.annot_cols).collect()
            phred_cate = np.array([[getattr(row, col) for col in slidingwindow.annot_cols] for row in annot_phred])
        else:
            phred_cate = None
        maf, is_rare, vset = prepare_vset_test(snps_mt_cate)
        vset_test.input_vset(vset, maf, is_rare, phred_cate)

        log.info(f'Doing analysis for window ({start_loc}, {end_loc}) ({vset_test.n_variants} variants) ...')
        pvalues = vset_test.do_inference(slidingwindow.annot_name)
        window_pvalues[(start_loc, end_loc)] = {'n_variants': vset_test.n_variants, 'pvalues': pvalues}

    return window_pvalues


def format_output(window_pvalues, chr, start_loc, end_loc, n_variants, n_voxels):
    """
    organizing pvalues to a structured format

    Parameters:
    ------------
    window_pvalues: a pd.DataFrame of pvalues of the window
    chr: chromosome
    start_loc: start position of the window
    end_loc: end position of the window
    n_variants: #variants of the window
    n_voxels: #voxels of the image

    Returns:
    ---------
    output: a pd.DataFrame of pvalues with metadata

    """
    meta_data = pd.DataFrame({'INDEX': range(1, n_voxels+1), 
                              'CHR': chr,
                              'START': start_loc, 'END': end_loc,
                              'N_VARIANT': n_variants})
    output = pd.concat([meta_data, window_pvalues], axis=1)
    return output


def check_input(args, log):
    # required arguments
    if args.geno_mt is None:
        raise ValueError('--geno-mt is required')
    if args.null_model is None:
        raise ValueError('--null-model is required')
    if args.range is None:
        raise ValueError('--range is required')
    
    # required files must exist
    if not os.path.exists(args.geno_mt):
        raise FileNotFoundError(f"{args.geno_mt} does not exist")
    if not os.path.exists(args.null_model):
        raise FileNotFoundError(f"{args.null_model} does not exist")

    # optional arguments
    if args.window_length is not None and args.window_length <= 0:
        raise ValueError('--window-length should be greater than 0')
    elif args.window_length is None:
        args.window_length = 2000
        log.info(f"Set --window-length as default 2000")

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    
    if args.maf_min is not None:
        if args.maf_min > 0.5 or args.maf_min < 0:
            raise ValueError('--maf-min must be greater than 0 and less than 0.5')
    else:
        args.maf_min = 0

    if args.variant_type is None:
        args.variant_type = 'snv'
        log.info(f"Set --variant-type as default 'snv'")
    else:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {'snv', 'variant', 'indel'}:
            raise ValueError("--variant-type must be one of ('variant', 'snv', 'indel')")
    
    if args.maf_max is None:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")
    elif args.maf_max > 0.5 or args.maf_max <= 0 or args.maf_max <= args.maf_min:
        raise ValueError(('--maf-max must be greater than 0, less than 0.5, '
                          'and greater than --maf-min'))
    
    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError('--mac-thresh must be greater than 0')
    args.mac_thresh = int(args.mac_thresh)

    if args.use_annotation_weights is None:
        args.use_annotation_weights = False
        log.info(f"Set --use-annotation-weights as False")

    # process arguments
    try:
        start, end = args.range.split(',')
        start_chr, start_pos = [int(x) for x in start.split(':')]
        end_chr, end_pos = [int(x) for x in end.split(':')]
    except:
        raise ValueError('--range should be in this format: <CHR>:<POS1>,<CHR>:<POS2>')
    if start_chr != end_chr:
        raise ValueError((f'starting with chromosome {start_chr} '
                            f'while ending with chromosome {end_chr} '
                            'is not allowed'))
    if start_pos > end_pos:
        raise ValueError((f'starting with {start_pos} '
                            f'while ending with position is {end_pos} '
                            'is not allowed'))

    temp_path = 'temp'
    i = 0
    while os.path.exists(temp_path + str(i)):
        i += 1
    temp_path += str(i)

    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference genome.')

    return start_chr, start_pos, end_pos, temp_path, geno_ref


def run(args, log):
    # checking if input is valid
    chr, start, end, temp_path, geno_ref = check_input(args, log)

    # reading data for unrelated subjects
    log.info(f'Read null model from {args.null_model}')
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        ids = file['id'][:].astype(str)
        bases = file['bases'][:]

    # subset voxels
    if args.voxel is not None:
        if np.max(args.voxel) + 1 <= bases.shape[0] and np.min(args.voxel) >= 0:
            log.info(f'{len(args.voxel)} voxels included.')
        else:
            raise ValueError('--voxel index (one-based) out of range')
    else:
        args.voxel = np.arange(bases.shape[0])
    bases = bases[args.voxel]

    # keep selected LDRs
    if args.n_ldrs is not None:
        resid_ldr, bases = keep_ldrs(args.n_ldrs, resid_ldr, bases)
        log.info(f'Keep the top {args.n_ldrs} LDRs and bases.')
        
    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None
    common_ids = get_common_ids(ids, keep_idvs)

    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs in --extract.")
    else:
        keep_snps = None

    # read genotype data
    spark_conf = {
    'spark.executor.memory': '8g',
    'spark.driver.memory': '8g',
    'spark.master': 'local[8]'
    }
    hl.init(quiet=True, spark_conf=spark_conf)
    hl.default_reference = geno_ref

    log.info(f'Reading genotype data from {args.geno_mt}')
    gprocessor = GProcessor.read_matrix_table(args.geno_mt, geno_ref=geno_ref, 
                                              variant_type=args.variant_type,
                                              maf_min=args.maf_min, maf_max=args.maf_max)
    
    # do preprocessing
    log.info(f"Processing genotype data ...")
    gprocessor.extract_snps(keep_snps)
    gprocessor.extract_idvs(common_ids)
    gprocessor.do_processing(mode='wgs')
    gprocessor.extract_gene(chr=chr, start=start, end=end)
    
    # save processsed data for faster analysis
    if not args.not_save_genotype_data:
        log.info(f'Save preprocessed genotype data to {temp_path}')
        gprocessor.save_interim_data(temp_path)

    try:
        gprocessor.check_valid()
        # extract and align subjects with the genotype data
        snps_mt_ids = gprocessor.subject_id()
        idx_common_ids = extract_align_subjects(ids, snps_mt_ids)
        resid_ldr = resid_ldr[idx_common_ids]
        covar = covar[idx_common_ids]
        covar = remove_dependent_columns(covar)
        log.info(f'{len(idx_common_ids)} common subjects in the data.')
        log.info(f"{covar.shape[1]} fixed effects in the covariates after removing redundant effects.\n")

        # single gene analysis
        vset_test = VariantSetTest(bases, resid_ldr, covar)
        window_pvalues = single_gene_analysis(
            gprocessor.snps_mt, chr, start, end, 
            args.variant_type, args.window_length, 
            args.use_annotation_weights, vset_test, log
        )
        
        # format output
        n_voxels = bases.shape[0]
        log.info('')
        for (start_loc, end_loc), window_results in window_pvalues.items():
            window_output = format_output(window_results['pvalues'], chr, start_loc, end_loc,
                                          window_results['n_variants'], n_voxels)
            out_path = f'{args.out}_chr{chr}_start{start_loc}_end{end_loc}_slidingwindow{args.window_length}.txt'
            window_output.to_csv(out_path, sep='\t', header=True, na_rep='NA', 
                                 index=None, float_format='%.5e')
            log.info(f'Save results for chr{chr} ({start_loc}, {end_loc}) to {out_path}')
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')