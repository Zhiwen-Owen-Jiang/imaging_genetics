import h5py
import logging
import numpy as np
import hail as hl
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.utils import *
from hail.linalg import BlockMatrix
from scipy.sparse import csr_matrix, coo_matrix


"""
Generate rare variant summary statistics:
1. LDR score statistics: U = Z'(I-M)\Xi, where relatedness has been removed from \Xi
2. Spark matrix of inner product of Z: Z'Z
3. low-dimensional format of Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
4. variance of each voxel under the null model: sigma^2

Allow to input a chromosome and save the summary statistics

"""


class RV:
    """
    Computing summary statistics for rare variants

    """

    def __init__(self, bases, resid_ldr, covar, locus, loco_preds=None):
        """
        Parameters:
        ------------
        bases: (N, r) np.array, functional bases
        resid_ldr: (n, r) np.array, LDR residuals
        covar: (n, p) np.array, the same as those used to do projection
        locus: hail.Table including locus, maf, is_rare, grch37, variant_type, 
                chr, start, and end
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects

        """
        # extract chr
        self.locus = locus
        chr = self.locus.aggregate(hl.agg.take(self.locus.locus.contig, 1)[0])
        if locus.locus.dtype.reference_genome.name == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)
        
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

        self.resid_ldr = resid_ldr.astype(np.float32)
        self.var = var.astype(np.float32)
        self.half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        self.bases = bases.astype(np.float32)
        self.n_subs = covar.shape[0]
        
    def sumstats(self, vset, bandwidth=5000):
        """
        Computing and writing summary statistics for rare variants

        Parameters:
        ------------
        vset: (m, n) csr_matrix of genotype

        """
        # n_variants and locus
        self.n_variants = vset.shape[0]

        # half ldr score
        self.half_ldr_score = vset @ self.resid_ldr # Z'(I-M)\Xi, (m, r)

        # Z'Z
        vset_ld = vset @ vset.T  # Z'Z, sparse matrix (m, m)
        banded_vset_ld = self._make_sparse_banded(vset_ld, bandwidth)
        self.bandwidth = bandwidth
        self.vset_ld_diag, self.vset_ld_data, self.vset_ld_row, self.vset_ld_col = banded_vset_ld
        self.vset_ld_shape = vset_ld.shape

        # Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
        self.vset_half_covar_proj = vset @ self.half_covar_proj # Z'UV', (m, p)
        
    @staticmethod
    def _make_sparse_banded(matrix, bandwidth):
        """
        Convert a sparse matrix into a sparse banded matrix.

        Parameters:
        ------------
        matrix (csr_matrix): sparse matrix.
        bandwidth (int): bandwidth around the diagonal to retain.

        Returns:
        ---------
        banded_matrix: Sparse banded matrix.

        """
        if not isinstance(matrix, csr_matrix):
            raise ValueError("Input matrix must be a csr_matrix.")
        
        row, col = matrix.nonzero() # both are np.array
        data = matrix.data
        
        diagonal_data = data[row == col]
        mask = (np.abs(row - col) <= bandwidth) & (col > row)
        banded_row = row[mask]
        banded_col = col[mask]
        banded_data = data[mask]
        
        return diagonal_data, banded_data, banded_row, banded_col

    def save(self, output_dir):
        """
        Saving summary statistics
        
        """
        with h5py.File(f"{output_dir}_rv_sumstats.h5", 'w') as file:
            file.create_dataset('bases', data=self.bases, dtype='float32')
            file.create_dataset('var', data=self.var, dtype='float32')
            file.create_dataset('half_ldr_score', data=self.half_ldr_score, dtype='float32')
            file.create_dataset('vset_half_covar_proj', data=self.vset_half_covar_proj, dtype='float32')
            file.create_dataset('vset_ld_diag', data=self.vset_ld_diag, dtype='float32')
            file.create_dataset('vset_ld_data', data=self.vset_ld_data, dtype='float32')
            file.create_dataset('vset_ld_row', data=self.vset_ld_row)
            file.create_dataset('vset_ld_col', data=self.vset_ld_col)
            file.create_dataset('vset_ld_shape', data=np.array(self.vset_ld_shape))
            file.attrs['n_subs'] = self.n_subs
            file.attrs['n_variants'] = self.n_variants
            file.attrs['bandwidth'] = self.bandwidth
        self.locus.write(f'{output_dir}_locus_info.ht', overwrite=True)


class RVsumstats:
    """
    Reading and processing WGS summary statistics for analysis

    """

    def __init__(self, prefix):
        self.locus = hl.read_table(f'{prefix}_locus_info.ht').key_by('locus', 'alleles')
        self.locus = self.locus.add_index('idx')
        self.geno_ref = self.locus.reference_genome.collect()[0]
        self.variant_type = self.locus.variant_type.collect()[0]
        self.maf = np.array(self.locus.maf.collect())
        self.is_rare = np.array(self.locus.is_rare.collect())

        with h5py.File(f"{prefix}_rv_sumstats.h5", 'r') as file:
            self.bases = file['bases'][:] # (N, r)
            self.var = file['var'][:] # (N, )
            self.half_ldr_score = file['half_ldr_score'][:]
            self.vset_half_covar_proj = file['vset_half_covar_proj'][:]
            self.n_variants = file.attrs['n_variants']
            self.n_subs = file.attrs['n_subs']
            self.bandwidth = file.attrs['bandwidth']

            vset_ld_diag = file['vset_ld_diag'][:]
            vset_ld_data = file['vset_ld_data'][:]
            vset_ld_row = file['vset_ld_row'][:]
            vset_ld_col = file['vset_ld_col'][:]
            vset_ld_shape = file['vset_ld_shape'][:]

        self.vset_ld = self._reconstruct_vset_ld(
            vset_ld_diag, vset_ld_data,
            vset_ld_row, vset_ld_col, vset_ld_shape
        )

        self.voxel_idxs = np.arange(self.bases.shape[0])
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _reconstruct_vset_ld(diag, data, row, col, shape):
        """
        Reconstructing a sparse matrix.

        Parameters:
        ------------
        diag: diagonal data
        data: banded data
        row: row index of upper band
        col: col index of lower band
        shape: sparse matrix shape

        Returns:
        ---------
        full_matrix: sparse banded matrix.

        """
        lower_row = col
        lower_col = row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([row, lower_row, diag_row_col])
        full_col = np.concatenate([col, lower_col, diag_row_col])
        full_data = np.concatenate([data, data, diag])
        
        full_matrix = csr_matrix((full_data, (full_row, full_col)), shape=shape)

        return full_matrix
        
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
            
    def extract_exclude_locus(self, extract_locus, exclude_locus):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
        exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format

        """
        if extract_locus is not None:
            extract_locus = parse_locus(extract_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(extract_locus.contains(self.locus.locus))
        if exclude_locus is not None:
            exclude_locus = parse_locus(exclude_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(~exclude_locus.contains(self.locus.locus))

    def extract_chr_interval(self, chr_interval=None):
        """
        Extacting a chr interval

        Parameters:
        ------------
        chr_interval: chr interval to extract

        """
        if chr_interval is not None:
            chr, start, end = parse_interval(chr_interval, self.geno_ref)
            interval = hl.locus_interval(chr, start, end, reference_genome=self.geno_ref)
            self.locus = self.locus.filter(interval.contains(self.locus.locus))
    
    def extract_maf(self, maf_min=None, maf_max=None):
        """
        Extracting variants by MAF 
        
        """
        if maf_min is not None:
            self.locus = self.locus.filter(self.locus.maf > maf_min)
        if maf_max is not None:
            self.locus = self.locus.filter(self.locus.maf <= maf_max)

    def annotate(self, annot):
        """
        Annotating functional annotations to locus
        ensuring no NA in annotations
        
        """
        self.locus = self.locus.annotate(annot=annot[self.locus.key])
        self.locus = self.locus.filter(hl.is_defined(self.locus.annot))
        self.locus = self.locus.cache()

    def get_interval(self):
        all_locus = self.locus.locus.collect()
        
        self.n_variants = len(all_locus) 
        if self.n_variants == 0:
            raise ValueError("no variant overlapping in summary statistics and annotations")
        else:
            self.logger.info(f"{self.n_variants} variants overlapping in summary statistics and annotations.")

        chr = all_locus[0].contig
        start = all_locus[0].position
        end = all_locus[-1].position
        
        return chr, start, end

    def parse_data(self, numeric_idx):
        """
        Parsing data for analysis

        Parameters:
        ------------
        numeric_idx: a list of numeric indices, 
        which are collected from `idx` in self.locus after filtering by annotations.

        Returns:
        ---------
        half_ldr_score: Z'(I-M)\Xi
        cov_mat: Z'(I-M)Z
        maf: a np.array of MAF
        is_rare: a np.array of boolean indices indicating MAC < mac_threshold
        
        """ 
        if max(numeric_idx) - min(numeric_idx) > self.bandwidth:
            raise ValueError('the variant set has exceeded the bandwidth of LD matrix')
        half_ldr_score = np.array(self.half_ldr_score[numeric_idx])
        vset_half_covar_proj = np.array(self.vset_half_covar_proj[numeric_idx])
        vset_ld = np.array(self.vset_ld[numeric_idx, numeric_idx])
        cov_mat = np.array((vset_ld - vset_half_covar_proj @ vset_half_covar_proj.T))
        maf = self.maf[numeric_idx]
        is_rare = self.is_rare[numeric_idx]

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
    vset: (m, n) csr_matrix of genotype

    """
    locus = snps_mt.rows().key_by().select('locus', 'alleles', 'maf', 'is_rare')
    locus = locus.annotate_globals(reference_genome=locus.locus.dtype.reference_genome.name)
    locus = locus.annotate_globals(variant_type=variant_type)
    bm = BlockMatrix.from_entry_expr(
        snps_mt.flipped_n_alt_alleles, mean_impute=True
    )
    if bm.shape[0] == 0 or bm.shape[1] == 0:
        raise ValueError("no variant in the genotype data")
    
    entries = bm.entries()
    non_zero_entries = entries.filter(entries.entry > 0)
    non_zero_entries = non_zero_entries.collect()
    rows = [entry['i'] for entry in non_zero_entries]
    cols = [entry['j'] for entry in non_zero_entries]
    values = [entry['entry'] for entry in non_zero_entries]

    vset = coo_matrix((values, (rows, cols)), shape=bm.shape, dtype=np.float32)
    vset = vset.tocsr()

    return vset, locus


def check_input(args, log):
    # required arguments
    if args.geno_mt is None and args.vcf is None and args.bfile is None:
        raise ValueError("one of --geno-mt, --vcf, or --bfile is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")

    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")

    if args.maf_max is None:
        if args.maf_min is not None and args.maf_min < 0.01 or args.maf_min is None:
            args.maf_max = 0.01
            log.info(f"Set --maf-max as default 0.01")

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    
    if args.bandwidth is None:
        args.bandwidth = 5000
        log.info("Set --bandwidth as default 5000")
    elif args.bandwidth <= 1:
        raise ValueError("--bandwidth must be greater than 1")


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
            if args.n_ldrs is not None:
                loco_preds.select_ldrs((0, args.n_ldrs))
            if loco_preds.ldr_col[1] - loco_preds.ldr_col[0] != null_model.n_ldrs:
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

        log.info('Preparing data ...')
        vset, locus = prepare_vset(gprocessor.snps_mt, gprocessor.variant_type)
        log.info(f'Computing summary statistics for {vset.shape[0]} variants ...\n') 
        process_wgs = RV(null_model.bases, null_model.resid_ldr, null_model.covar, locus, loco_preds)
        process_wgs.sumstats(vset, args.bandwidth)
        process_wgs.save(args.out)

        log.info((f'Save summary statistics to\n'
                  f'{args.out}_rv_sumstats.h5\n'
                  f'{args.out}_locus_info.ht'))

    finally:
        if 'loco_preds' in locals():
            loco_preds.close()