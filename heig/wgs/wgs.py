import h5py
import logging
import numpy as np
import hail as hl
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.utils import *
from hail.linalg import BlockMatrix


"""
Generate rare variant summary statistics:
1. LDR score statistics: U = Z'(I-M)\Xi, where relatedness has been removed from \Xi
2. Spark matrix of inner product of Z: Z'Z
3. low-dimensional format of Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
4. variance of each voxel under the null model: sigma^2

Allow to input a chromosome and save the summary statistics

TODO:
1. test if save Z'Z for the entire chr is too big 

"""


class RV:
    """
    Computing summary statistics for rare variants

    """

    def __init__(self, bases, resid_ldr, covar, loco_preds=None):
        """
        Parameters:
        ------------
        bases: (N, r) np.array, functional bases
        resid_ldr: (n, r) np.array, LDR residuals
        covar: (n, p) np.array, the same as those used to do projection
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects

        """
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
        self.n_subs = covar.shape[0]
        
    def sumstats(self, vset, locus):
        """
        Computing and writing summary statistics for rare variants

        Parameters:
        ------------
        vset: (m, n) blockmatrix of genotype data
        locus: hail.Table including locus, maf, is_rare, grch37, variant_type, 
                chr, start, and end

        """
        # n_variants and locus
        self.n_variants = vset.shape[0]
        self.locus = locus

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
            file.attrs['n_subs'] = self.n_subs
            file.attrs['n_variants'] = self.n_variants

        self.half_ldr_score.write(f"{output_dir}_half_ldr_score.bm")
        self.inner_vset.write(f"{output_dir}_vset_ld.bm")
        self.vset_half_covar_proj.write(f"{output_dir}_vset_half_covar.bm")
        self.locus.write(f'{output_dir}_locus_info.ht', overwrite=True)


class RVsumstats:
    """
    Reading and processing WGS summary statistics for analysis

    """

    def __init__(self, prefix):
        self.half_ldr_score = BlockMatrix.read(f"{prefix}_half_ldr_score.bm") # (m, r)
        self.inner_vset = BlockMatrix.read(f"{prefix}_vset_ld.bm") # (m, m)
        self.vset_half_covar_proj = BlockMatrix.read(f"{prefix}_vset_half_covar.bm") # (m, p)
        self.locus = hl.read_table(f'{prefix}_locus_info.ht').key_by('locus', 'alleles')
        self.locus = self.locus.add_index('idx')
        self.geno_ref = self.locus.reference_genome.collect()[0]
        self.variant_type = self.locus.variant_type.collect()[0]

        with h5py.File(f"{self.output_dir}_misc.h5", 'r') as file:
            self.bases = file['bases'][:] # (N, r)
            self.var = file['var'][:] # (N, )
            self.n_variants = file.attrs['n_variants']
            self.n_subs = file.attrs['n_subs']

        self.variant_idxs = None
        self.voxel_idxs = np.arange(self.bases.shape[0])
        self.logger = logging.getLogger(__name__)

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.bases.shape[1] and n_ldrs <= self.half_ldr_score.shape[1]:
                self.bases = self.bases[:, :n_ldrs]
                self.half_ldr_score = self.half_ldr_score[:, :n_ldrs]
                self.logger.info(f"Keep the top {n_ldrs} LDRs.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs")

    def select_voxels(self, voxel_idxs=None):
        if voxel_idxs is not None:
            if np.max(voxel_idxs) < self.bases.shape[0]:
                self.voxel_idxs = voxel_idxs
                self.bases = self.bases[voxel_idxs]
                self.logger.info(f"{len(voxel_idxs)} voxels included.")
            else:
                raise ValueError("--voxel index (one-based) out of range")
            
    def select_variants(self):
        """
        Selecting variants from summary statistics
        
        """
        if self.n_variants < self.half_ldr_score.shape[0]:
            self.variant_idxs = self.locus.idx.collect()
            self.half_ldr_score = self.half_ldr_score.filter_rows(self.variant_idxs)
            self.inner_vset = self.inner_vset.filter(self.variant_idxs, self.variant_idxs)
            self.vset_half_covar_proj = self.vset_half_covar_proj.filter_rows(self.variant_idxs)

    def extract_exclude_locus(self, extract_locus, exclude_locus):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
        exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format

        """
        if extract_locus is not None:
            # extract_locus = hl.Table.from_pandas(extract_locus[["locus"]])
            # extract_locus = extract_locus.annotate(locus=hl.parse_locus(extract_locus.locus))
            extract_locus = parse_locus(extract_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(extract_locus.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --extract-locus.")
        if exclude_locus is not None:
            # exclude_locus = hl.Table.from_pandas(exclude_locus[["locus"]])
            # exclude_locus = exclude_locus.annotate(locus=hl.parse_locus(exclude_locus.locus))
            exclude_locus = parse_locus(exclude_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(~exclude_locus.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --exclude-locus.")

    def extract_chr_interval(self, chr_interval=None):
        """
        Extacting a chr interval

        Parameters:
        ------------
        chr_interval: chr interval to extract

        """
        if chr_interval is not None:
            chr, start, end = process_range(chr_interval)
            interval = hl.locus_interval(chr, start, end, reference_genome=self.geno_ref)
            self.locus = self.locus.filter(interval.contains(self.locus.locus))
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after --chr-interval.")
    
    def extract_maf(self, maf_min=None, maf_max=None):
        """
        Extracting variants by MAF 
        
        """
        if maf_min is not None:
            self.locus = self.locus.filter(self.locus.maf < maf_min)
        if maf_max is not None:
            self.locus = self.locus.filter(self.locus.maf >= maf_max)
        if maf_min is not None or maf_max is not None:
            self.n_variants = self.locus.count()
            self.logger.info(f"{self.n_variants} variants remaining after filtering by MAF.")

    def semi_join(self, annot):
        """
        Semi join with annotations
        
        """
        self.locus = self.locus.semi_join(annot)
        self.n_variants = self.locus.count()

    def parse_data(self, idx):
        """
        Parse data for analysis, must do select_variants() before

        Parameters:
        ------------
        idx: hail.expr of numeric indices of mask
        
        """ 
        half_ldr_score = self.half_ldr_score.filter_rows[idx].to_numpy()
        vset_half_covar_proj = self.vset_half_covar_proj.filter_rows[idx]
        inner_vset = self.inner_vset.filter[idx, idx]
        cov_mat = (inner_vset - vset_half_covar_proj @ vset_half_covar_proj.T).to_numpy()
        locus = self.locus.filter(idx)
        maf = np.array(locus.maf.collect())
        is_rare = np.array(locus.is_rare.collect())

        return half_ldr_score, cov_mat, maf, is_rare


def prepare_vset(snps_mt, variant_type):
    """
    Extracting data from MatrixTable

    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype data
    variant_type: variant type

    Returns:
    ---------
    vset: (m, n) BlockMatrix

    """
    locus = snps_mt.rows().key_by().select('locus', 'alleles', 'maf', 'is_rare')
    locus = locus.annotate_globals(reference_genome=locus.locus.dtype.reference_genome.name)
    locus = locus.annotate_globals(variant_type=variant_type)
    vset = BlockMatrix.from_entry_expr(
        snps_mt.flipped_n_alt_alleles, mean_impute=True
    )
    return vset, locus


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


def run(args, log):
    # checking if input is valid
    check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting LDRs
    log.info(f"Read null model from {args.null_model}")
    null_model = NullModel(args.null_model)
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
        gprocessor = read_genotype_data(args, log)
    
        # do preprocessing
        log.info(f"Processing genetic data ...")
        gprocessor.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        gprocessor.extract_chr_interval(args.chr_interval)
        gprocessor.keep_remove_idvs(common_ids)
        gprocessor.do_processing(mode="wgs")
        gprocessor.check_valid()

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
        vset, locus = prepare_vset(gprocessor.snps_mt, gprocessor.variant_type) 
        process_wgs = RV(null_model.bases, null_model.resid_ldr, null_model.covar, loco_preds)
        process_wgs.sumstats(vset, locus)
        process_wgs.save(args.out)

        log.info((f'Save summary statistics to\n'
                  f'{args.out}_half_ldr_score.bm\n'
                  f'{args.out}_inner_vset.bm\n'
                  f'{args.out}_vset_U.bm\n'
                  f'{args.out}_misc.h5\n'
                  f'{args.out}_locus_info.ht'))

    finally:
        if 'loco_preds' in locals():
            loco_preds.close()