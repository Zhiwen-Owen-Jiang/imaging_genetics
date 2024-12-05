import os
import h5py
import shutil
import logging
import numpy as np
import hail as hl
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.utils import *
from hail.linalg import BlockMatrix


"""
Generate WGS summary statistics:
1. LDR score statistics: U = Z'(I-M)\Xi, where relatedness has been removed from \Xi
2. Spark matrix of inner product of Z: Z'Z
3. low-dimensional format of Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
4. variance of each voxel under the null model: sigma^2

Allow to input a chromosome and save the summary statistics

TODO:
1. test if save Z'Z for the entire chr is too big 

"""


class ProcessWGS:
    """
    Computing summary statistics for rare variants

    """

    def __init__(self, bases, resid_ldr, covar, maf, is_rare, grch37, variant_type, 
                 chr, start, end, loco_preds=None):
        """
        Parameters:
        ------------
        bases: (N, r) np.array, functional bases
        resid_ldr: (n, r) np.array, LDR residuals
        covar: (n, p) np.array, the same as those used to do projection
        maf: (m, ) np.array, minor allele frequency
        is_rare: (m, ) np.array, if the minor allele count less than a threshold
        grch37: if genome reference is GRCh37
        variant_type: variant type
        chr: include which chr
        start: start bp
        end: end bp
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects

        """
        if grch37:
            self.geno_ref = "GRCh37"
        else:
            self.geno_ref = "GRCh38"
    
        # adjust for relatedness
        if loco_preds is not None:
            resid_ldr -= loco_preds.data_reader(chr) # (I-M)\Xi, (n, r)

        # variance
        inner_ldr = np.dot(resid_ldr.T, resid_ldr)  # \Xi'(I-M)\Xi, (r, r)
        var = np.sum(np.dot(bases, inner_ldr) * bases, axis=1) / (
            covar.shape[0] - covar.shape[1]
        )  # (N, ), save once

        # X = UDV', X'(X'X)^{-1/2} = UV'
        covar_U, _, covar_Vt = np.linalg.svd(covar, full_matrices=False)

        self.resid_ldr = BlockMatrix.from_numpy(resid_ldr)
        self.var = var
        self.half_covar_proj = BlockMatrix.from_numpy(np.dot(covar_U, covar_Vt).astype(np.float32))
        self.bases = bases
        self.maf = maf
        self.is_rare = is_rare
        self.variant_type = variant_type
        self.chr = chr
        self.start = start
        self.end = end
        self.n_subs = covar.shape[0]
        
    def sumstats(self, vset):
        """
        Computing and writing summary statistics for rare variants

        Parameters:
        ------------
        vset: (m, n) blockmatrix of genotype data

        """
        # n_variants
        self.n_variants = vset.shape[0]

        # half ldr score
        self.half_ldr_score = vset @ self.resid_ldr # Z'(I-M)\Xi, (m, r)

        # Z'Z
        inner_vset = vset @ vset.T  # Z'Z, sparse matrix (m, m)
        self.inner_vset = inner_vset.sparsify_band(-500000, 500000) # band size 500k

        # Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
        self.vset_half_covar_proj = vset @ self.half_covar_proj # Z'UV', (m, p)

    def save(self, output_dir):
        """
        Saving summary statistics
        
        """
        with h5py.File(f"{output_dir}_misc.h5", 'w') as file:
            file.create_dataset('bases', data=self.bases, dtype='float32')
            file.create_dataset('var', data=self.var, dtype='float32')
            file.create_dataset('maf', data=self.maf, dtype='float32')
            file.create_dataset('is_rare', data=self.is_rare)
            file.attrs['variant_type'] = self.variant_type
            file.attrs['chr'] = self.chr
            file.attrs['start'] = self.start
            file.attrs['end'] = self.end
            file.attrs['n_subs'] = self.n_subs
            file.attrs['n_variants'] = self.n_variants
            file.attrs['geno_ref'] = self.geno_ref

        self.half_ldr_score.write(f"{output_dir}_half_ldr_score.bm")
        self.inner_vset.write(f"{output_dir}_vset_ld.bm")
        self.vset_half_covar_proj.write(f"{output_dir}_vset_half_covar.bm")


class WGS:
    """
    Reading and processing WGS summary statistics

    """

    def __init__(self, prefix):
        self.half_ldr_score = BlockMatrix.read(f"{prefix}_half_ldr_score.bm") # (m, r)
        self.inner_vset = BlockMatrix.read(f"{prefix}_vset_ld.bm") # (m, m)
        self.vset_half_covar_proj = BlockMatrix.read(f"{prefix}_vset_half_covar.bm") # (m, p)

        with h5py.File(f"{self.output_dir}_misc.h5", 'r') as file:
            self.bases = file['bases'][:]
            self.var = file['var'][:]
            self.maf = file['maf'][:]
            self.is_rare = file['is_rare'][:]
            self.geno_ref = file.attrs['geno_ref']
            self.chr = file.attrs['chr']
            self.start = file.attrs['start']
            self.end = file.attrs['end']
            self.n_variants = file.attrs['n_variants']

        self.locus = hl.read_table(f'{prefix}_locus_info.ht').key_by('locus')
        self.locus = self.locus.add_index('idx')
        self.idxs = None
        self.logger = logging.getLogger(__name__)

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.bases.shape[1] and n_ldrs <= self.half_ldr_score.shape[1]:
                self.bases = self.bases[:, :n_ldrs]
                self.half_ldr_score = self.half_ldr_score[:, :n_ldrs]
                self.logger.info(f"Keep the top {n_ldrs} LDRs.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs")
            
    def select_variants(self):
        """
        Selecting variants from summary statistics
        
        """
        if self.idxs is None:
            return self.half_ldr_score, self.inner_vset, self.vset_half_covar_proj
        else:
            half_ldr_score = self.half_ldr_score.filter_rows(self.idxs)
            inner_vset = self.inner_vset.filter(self.idxs, self.idxs)
            vset_half_covar_proj = self.vset_half_covar_proj.filter_rows(self.idxs)

            return half_ldr_score, inner_vset, vset_half_covar_proj

    def extract_exclude_locus(self, extract_locus, exclude_locus):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a pd.DataFrame of SNPs in `chr:POS` format
        exclude_locus: a pd.DataFrame of SNPs in `chr:POS` format

        """
        if extract_locus is not None:
            extract_locus = hl.Table.from_pandas(extract_locus[["locus"]])
            extract_locus = extract_locus.annotate(locus=hl.parse_locus(extract_locus.locus))
            self.locus = self.locus.filter(extract_locus.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --extract-locus.")
        if exclude_locus is not None:
            exclude_locus = hl.Table.from_pandas(exclude_locus[["locus"]])
            exclude_locus = exclude_locus.annotate(locus=hl.parse_locus(exclude_locus.locus))
            self.locus = self.locus.filter(~exclude_locus.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --exclude-locus.")
        if extract_locus is not None or exclude_locus is not None:
            self.idxs = self.locus.idx.collect()

    def extract_chr_interval(self, start=None, end=None):
        """
        Extacting a chr interval

        Parameters:
        ------------
        start: start position
        end: end position

        """
        if start is not None and end is not None:
            interval = hl.locus_interval(self.chr, start, end, reference_genome=self.geno_ref)
            self.locus = self.locus.filter(interval.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --chr-interval-locus.")
            self.idxs = self.locus.idx.collect()
    
    def extract_maf(self, maf_min, maf_max):
        """
        Extracting variants by MAF 
        
        """
        self.idxs = list(np.where((maf_min < self.maf) & (self.maf >= maf_max))[0])


def prepare_vset(snps_mt):
    """
    Extracting data from MatrixTable

    Parameters:
    ------------
    snps_mt: MatrixTable

    Returns:
    ---------
    vset: (m, n) BlockMatrix
    maf: (m, ) np.array of MAF
    is_rare: (m, ) np.array boolean index indicating MAC < mac_threshold

    """
    maf = np.array(snps_mt.maf.collect())
    is_rare = np.array(snps_mt.is_rare.collect())
    vset = BlockMatrix.from_entry_expr(
        snps_mt.flipped_n_alt_alleles, mean_impute=True
    )
    return maf, is_rare, vset


def check_input(args, log):
    # required arguments
    if args.geno_mt is None and args.vcf is None and args.bfile is None:
        raise ValueError("one of --geno-mt, --vcf, or --bfile is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")

    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")

    if args.maf_max is None and args.maf_min < 0.01:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")

    # process arguments
    start_chr, start_pos, end_pos = process_range(args.chr_interval)

    return start_chr, start_pos, end_pos


def run(args, log):
    # checking if input is valid
    chr, start, end = check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting voxels and LDRs
    log.info(f"Read null model from {args.null_model}")
    null_model = NullModel(args.null_model)
    null_model.select_voxels(args.voxel)
    null_model.select_ldrs(args.n_ldrs)

    # read loco preds
    try:
        if args.loco_preds is not None:
            log.info(f"Read LOCO predictions from {args.loco_preds}")
            loco_preds = LOCOpreds(args.loco_preds)
            loco_preds.select_ldrs(args.n_ldrs)
            if loco_preds.n_ldrs != null_model.n_ldrs:
                raise ValueError(
                    (
                        "inconsistent dimension in LDRs and LDR LOCO predictions. "
                        "Try to use --n-ldrs"
                    )
                )
            common_ids = ds.get_common_idxs(
                null_model.ids,
                loco_preds.ids,
                args.keep
            )
        else:
            common_ids = ds.get_common_idxs(null_model.ids, args.keep, single_id=True)
        common_ids = ds.remove_idxs(common_ids, args.remove, single_id=True)

        # read genotype data
        if args.geno_mt is not None:
            log.info(f"Read MatrixTable from {args.geno_mt}")
            read_func = GProcessor.read_matrix_table
            data_path = args.geno_mt
        elif args.bfile is not None:
            log.info(f"Read bfile from {args.bfile}")
            read_func = GProcessor.import_plink
            data_path = args.bfile
        elif args.vcf is not None:
            log.info(f"Read VCF from {args.vcf}")
            read_func = GProcessor.import_vcf
            data_path = args.vcf

        gprocessor = read_func(
                data_path,
                grch37=args.grch37,
                hwe=args.hwe,
                variant_type=args.variant_type,
                maf_min=args.maf_min,
                maf_max=args.maf_max,
                mac_thresh=args.mac_thresh,
                call_rate=args.call_rate,
                chr=chr,
                start=start,
                end=end
        )
    
        # do preprocessing
        log.info(f"Processing genetic data ...")
        gprocessor.extract_exclude_snps(args.extract, args.exclude)
        gprocessor.keep_remove_idvs(common_ids)
        gprocessor.do_processing(mode="wgs")
        gprocessor.check_valid()

        if not args.not_save_genotype_data:
            temp_path = get_temp_path()
            gprocessor.save_interim_data(temp_path)

        if chr is None:
            chr, start, end = gprocessor.extract_range()
        # extract and align subjects with the genotype data
        snps_mt_ids = gprocessor.subject_id()
        null_model.keep(snps_mt_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(snps_mt_ids)} common subjects in the data.")
        log.info(
            (f"{null_model.covar.shape[1]} fixed effects in the covariates (including the intercept) "
             "after removing redundant effects.\n")
        )

        if args.loco_preds is not None:
            loco_preds.keep(snps_mt_ids)
        else:
            loco_preds = None

        log.info('Computing summary statistics ...\n')
        maf, is_rare, vset = prepare_vset(gprocessor.snps_mt)
        process_wgs = ProcessWGS(null_model.bases, null_model.resid_ldr, 
                                 null_model.covar, maf, is_rare, args.grch37,
                                 args.variant_type, chr, start, end, loco_preds)
        process_wgs.sumstats(vset)
        process_wgs.save(args.out)
        gprocessor.write_locus(args.out)

        log.info((f'Save summary statistics to\n'
                  f'{args.out}_half_ldr_score.bm\n'
                  f'{args.out}_inner_vset.bm\n'
                  f'{args.out}_vset_U.bm\n'
                  f'{args.out}_misc.h5\n'
                  f'{args.out}_locus_info.ht'))

    finally:
        if "temp_path" in locals():
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                log.info(f"Removed preprocessed genotype data at {temp_path}")
        if 'loco_preds' in locals():
            loco_preds.close()