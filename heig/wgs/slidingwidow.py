import os
import shutil
import h5py
import numpy as np
import hail as hl
from heig.wgs.staar import VariantSetTest
import heig.input.dataset as ds
from heig.wgs.utils import *



class SlidingWindow:
    def __init__(self, snps_mt, variant_type, window_length):
        """
        The specific type of variants have been extracted; genotype data has aligned with covariates
        
        """
        self.snps_mt = snps_mt
        self.variant_type = variant_type
        self.window_length = window_length
        self.anno_pred = self.get_annotation()
        self.windows = self.get_windows()


    def get_annotation(self):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        """
        if self.variant_type != 'snv':
            anno_phred = self.snps.fa.annotate(null_weight=1)
        else:
            anno_cols = [Annotation_name_catalog[anno_name]
                        for anno_name in Annotation_name]

            # anno_phred = self.snps.fa[anno_cols].to_pandas()
            # anno_phred['cadd_phred'] = anno_phred['cadd_phred'].fillna(0)
            # anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
            # anno_phred['apc_local_nucleotide_diversity2'] = anno_local_div
            
            anno_phred = self.snps.fa.select(*anno_cols)
            anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
            anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
            anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div)    
        return anno_phred
    
    def get_windows(self, start, end):
        windows = list()
        sliding_length = self.window_length // 2
        cur_left = start
        cur_right = start + sliding_length
        while cur_right <= end:
            windows.append(tuple([cur_left, cur_right]))
            cur_left, cur_right = cur_right, cur_right + sliding_length
        return windows
    

def single_gene_analysis(snps, start, end, variant_type, window_length, vset_test, log):
    """
    Single gene analysis

    Parameters:
    ------------
    snps: a MatrixTable of annotated vcf
    start: start position of gene
    end: end position of gene
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # extracting specific variant type and the gene
    snps = extract_variant_type(snps, variant_type)
    snps = extract_gene(start, end, snps)
    vset = fillna_flip_snps(snps.GT.to_numpy())
    phred = slidingwindow.anno_phred.to_numpy()

    # individual analysis
    window_pvalues = dict()
    slidingwindow = SlidingWindow(snps, variant_type, window_length)
    for start_loc, end_loc in slidingwindow.windows:
        phred_ = phred[start_loc: end_loc]
        vset_ = vset[:, start_loc: end_loc]
        vset_test.input_vset(vset_, phred_)
        pvalues = vset_test.do_inference()
        window_pvalues[(start_loc, end_loc)] = pvalues


def check_input(args, log):
    pass


def run(args, log):
    # checking if input is valid
    chr, start, end, voxel_list, variant_category, temp_path, geno_ref = check_input(args, log)

    # reading data for unrelated subjects
    log.info(f'Read null model from {args.null_model}')
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        ids = file['id'][:].astype(str)
        bases = file['bases'][:]

    # subset voxels
    if voxel_list is not None:
        if np.max(voxel_list) + 1 <= bases.shape[0] and np.min(voxel_list) >= 0:
            log.info(f'{len(voxel_list)} voxels included.')
        else:
            raise ValueError('--voxel index (one-based) out of range')
    else:
        voxel_list = np.arange(bases.shape[0])
    bases = bases[voxel_list]

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
        cate_pvalues = single_gene_analysis(gprocessor.snps_mt, start, end, 
                                            args.window_length, vset_test, log)
        
        # format output
        n_voxels = bases.shape[0]
        log.info('')
        for cate, cate_results in cate_pvalues.items():
            cate_output = format_output(cate_results['pvalues'], chr, start, end,
                                        cate_results['n_variants'], n_voxels, cate)
            out_path = f'{args.out}_chr{chr}_start{start}_end{end}_{cate}.txt'
            cate_output.to_csv(out_path, sep='\t', header=True, na_rep='NA', 
                               index=None, float_format='%.5e')
            log.info(f'Save results for {OFFICIAL_NAME[cate]} to {out_path}')
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')



    vset_test = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    snps = hl.read_matrix_table(args.avcfmt)

    # extracting common ids
    snps_ids = set(snps.s.collect())
    common_ids = snps_ids.intersection(ids)
    snps = snps.filter_cols(hl.literal(common_ids).contains(snps.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis (do parallel)
    res = single_gene_analysis(snps, start, end, args.window_length, vset_test)

   