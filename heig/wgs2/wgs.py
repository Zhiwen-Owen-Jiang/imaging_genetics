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

"""


class ProcessWGS:
    """
    Computing summary statistics for rare variants

    """

    def __init__(self, bases, resid_ldr, covar, maf, is_rare, geno_ref, output_dir, chr, loco_preds=None):
        """
        Parameters:
        ------------
        bases: (N, r) np.array, functional bases
        resid_ldr: (n, r) np.array, LDR residuals
        covar: (n, p) np.array, the same as those used to do projection
        maf: (m, ) np.array, minor allele frequency
        is_rare: (m, ) np.array, if the minor allele count less than a threshold
        geno_ref: genome reference
        output_dir: a directory to save summary statistics
        chr: include which chr
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

        # X = UDV'
        covar_U, _, covar_Vt = np.linalg.svd(covar, full_matrices=False)
        covar_V = covar_Vt.T # V, (n, p), save once

        self.resid_ldr = BlockMatrix.from_numpy(resid_ldr)
        self.var = var
        self.covar_U = BlockMatrix.from_numpy(covar_U)
        self.covar_V = covar_V
        self.output_dir = output_dir
        
        with h5py.File(f"{self.output_dir}_misc.h5", 'w') as file:
            file.create_dataset('bases', data=bases, dtype='float32')
            file.create_dataset('var', data=self.var, dtype='float32')
            file.create_dataset('covar_V', data=self.covar_V, dtype='float32')
            file.create_dataset('maf', data=maf, dtype='float32')
            file.create_dataset('is_rare', data=is_rare)
            file.attrs['chr'] = chr
            file.attrs['n_subs'] = covar.shape[0]
            file.attrs['geno_ref'] = geno_ref

    def sumstats(self, vset):
        """
        Computing and writing summary statistics for rare variants

        Parameters:
        ------------
        vset: (n, m) blockmatrix of genotype data

        """
        # half ldr score
        half_ldr_score = vset @ self.resid_ldr # Z'(I-M)\Xi, (m, r)
        half_ldr_score.write(f"{self.output_dir}_half_ldr_score.bm")

        # Z'Z
        inner_vset = vset @ vset.T  # Z'Z, sparse matrix (m, m)
        inner_vset = inner_vset.sparsify_band(-500000, 500000) # band size 500k
        inner_vset.write(f"{self.output_dir}_inner_vset.bm")

        # Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
        vset_U = vset @ self.covar_U # Z'U, (m, p)
        vset_U.write(f"{self.output_dir}_vset_U.bm")


class WGS:
    """
    Reading WGS summary statistics

    """

    def __init__(self, prefix):
        self.half_ldr_score = BlockMatrix.read(f"{prefix}_half_ldr_score.bm") # (m, r)
        self.inner_vset = BlockMatrix.read(f"{prefix}_inner_vset.bm") # (m, m)
        self.vset_U = BlockMatrix.read(f"{prefix}_vset_U.bm") # (m, p)

        with h5py.File(f"{self.output_dir}_misc.h5", 'r') as file:
            self.bases = file['bases'][:]
            self.var = file['var'][:]
            self.covar_V = file['covar_V'][:]
            self.maf = file['maf'][:]
            self.is_rare = file['is_rare'][:]
            self.geno_ref = file.attrs['geno_ref']
            self.chr = file.attrs['chr']

        self.locus = hl.read_table(f'{prefix}_locus_info.ht').key_by('rsID')
        self.locus = self.locus.add_index('idx')
        self.logger = logging.getLogger(__name__)

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.bases.shape[1] and n_ldrs <= self.half_ldr_score.shape[1]:
                self.bases = self.bases[:, :n_ldrs]
                self.half_ldr_score = self.half_ldr_score[:, :n_ldrs]
                self.logger.info(f"Keep the top {n_ldrs} LDR LOCO predictions.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs in LOCO predictions")
            
    def _select_varints(self, variant_idxs):
        """
        variant_idxs: int or list, must be non-empty and increasing
        
        """
        if len(variant_idxs) == 0:
            raise ValueError('variant indices must be non-empty')
        variant_idxs = sorted(list(variant_idxs))

        half_ldr_score = self.half_ldr_score.filter_rows(variant_idxs)
        inner_vset = self.inner_vset.filter(variant_idxs, variant_idxs)
        vset_U = self.vset_U.filter_rows(variant_idxs)

        return half_ldr_score, inner_vset, vset_U

    def extract_snps(self, keep_snps):
        """
        Extracting variants by rsID

        Parameters:
        ------------
        keep_snps: a pd.DataFrame of SNPs

        """
        rs_ht = hl.Table.from_pandas(keep_snps["SNP"], key='rsID')
        filtered_with_index = self.locus.semi_join(rs_ht)
        indices = filtered_with_index.idx.collect()

        return self._select_varints(indices)

    def extract_region(self, start, end):
        """
        Extacting variants by a region

        Parameters:
        ------------
        start: start position
        end: end position

        """
        region = hl.locus_interval(self.chr, start, end, reference_genome=self.geno_ref)
        filtered_with_index = self.locus.filter(region.contains(self.locus.locus))
        indices = filtered_with_index.idx.collect()

        return self._select_varints(indices)
    
    def extract_maf(self, maf_min, maf_max):
        indices = np.where((maf_min < self.maf) & (self.maf >= maf_max))[0]
        return self._select_varints(indices)


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
    if args.geno_mt is None:
        raise ValueError("--geno-mt is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")

    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")
    else:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {"snv", "variant", "indel"}:
            raise ValueError(
                "--variant-type must be one of ('variant', 'snv', 'indel')"
            )

    if args.maf_max is None:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")
    elif args.maf_max > 0.5 or args.maf_max <= 0 or args.maf_max <= args.maf_min:
        raise ValueError(
            (
                "--maf-max must be greater than 0, less than 0.5, "
                "and greater than --maf-min"
            )
        )

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    args.mac_thresh = int(args.mac_thresh)

    # process arguments
    if args.range is not None:
        start_chr, start_pos, end_pos = process_range(args.range)
    else:
        start_chr, start_pos, end_pos =  None, None, None

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
                args.keep,
                single_id=True,
            )
        else:
            common_ids = ds.get_common_idxs(null_model.ids, args.keep, single_id=True)

        # read genotype data
        log.info(f"Reading genotype data from {args.geno_mt}")
        gprocessor = GProcessor.read_matrix_table(
            args.geno_mt,
            grch37=args.grch37,
            variant_type=args.variant_type,
            hwe=args.hwe,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            mac_thresh=args.mac_thresh,
            call_rate=args.call_rate,
        )

        # do preprocessing
        log.info(f"Processing genotype data ...")
        gprocessor.extract_snps(args.extract)
        gprocessor.extract_idvs(common_ids)
        gprocessor.do_processing(mode="wgs")
        gprocessor.extract_gene(chr=chr, start=start, end=end)

        # save processsed data for faster analysis
        if not args.not_save_genotype_data:
            temp_path = get_temp_path()
            log.info(f"Save preprocessed genotype data to {temp_path}")
            gprocessor.save_interim_data(temp_path)

        gprocessor.check_valid()
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
                                 null_model.covar, maf, is_rare, 
                                 args.out, chr, loco_preds)
        process_wgs.sumstats(vset)
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
        if args.loco_preds is not None:
            loco_preds.close()
