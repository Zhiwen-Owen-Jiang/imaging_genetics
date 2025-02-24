import time
import logging
import numpy as np
import hail as hl
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.sparse import csr_matrix
import heig.input.dataset as ds
from heig.wgs.null import NullModel
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.mt import SparseGenotype
from heig.wgs.wgs2 import SparseBandedLD
from heig.wgs.coding import Coding
from heig.utils import find_loc
from heig.wgs.utils import *


"""
cluster inference

1. generate a bootstrap sample from null model
2. compute rv summary statistics
3. split summary statistics into random sets, permute sets?
4. do analysis using STAAR
5. count cluster size

input:
1. null model
2. sparse genotype
3. sig threshold
4. number of bootstrap samples

output:
1. null distribution of cluster size 

TODO:
1. choose other test: SKAT/burden
2. as of (v1.3.0), cluster analysis only supports FAVOR annot and snv
    
"""


class RVcluster:
    """
    Computing cluster size (i.e., number of associated voxels)
    for variant sets in a bootstrap sample

    1. compute rare variant summary statistics
    2. sliding window analysis
    3. count cluster size for each sample-variant-set pair for multiple tests

    """

    def __init__(
            self, 
            null_model, 
            vset,
            chr, 
            gene_numeric_idxs,
            phred_cate,
            annot_name,
            maf,
            mac,
            sig_thresh=2.5e-6, 
            mac_thresh=10,
            cmac_min=2,
            threads=1,
            loco_preds=None,
            voxels=None
        ):
        """
        Parameters:
        ------------
        null_model: a NullModel instance
        vset: (m, n) csr_matrix of genotype
        chr: chromosome
        gene_numeric_idxs: a list of variant idxs for genes
        phred_cate: functional annotations
        annot_name: names of annotations
        maf: a np.array of MAF
        mac: a np.array of MAC
        sig_thresh: significant threshold
        mac_thresh: a MAC threshold to denote ultrarare variants for ACAT-V
        cmac_min: the minimal cumulative MAC for a variant set
        threads: number of threads
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects
        voxels: a np.array of voxel idxs (0-based), may be None

        """
        # self.null_model = null_model
        self.bases = null_model.bases.astype(np.float32)
        self.vset = vset
        self.numeric_idx_list = gene_numeric_idxs
        self.phred_cate = phred_cate
        self.annot_name = annot_name
        self.maf = maf
        self.mac = mac
        self.sig_thresh = sig_thresh
        self.mac_thresh = mac_thresh
        self.cmac_min = cmac_min
        self.threads = threads
        self.n_variants, self.n_subs = self.vset.shape
        self.n_covars = null_model.covar.shape[1]
        self.logger = logging.getLogger(__name__)
        
        covar_U, _, covar_Vt = np.linalg.svd(null_model.covar, full_matrices=False)
        half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        self.vset_half_covar_proj = self.vset @ half_covar_proj

        if loco_preds is not None:
            # extract chr
            # chr = self.mask.annot.aggregate(hl.agg.take(self.mask.annot.locus.contig, 1)[0])
            # if self.mask.annot.locus.dtype.reference_genome.name == "GRCh38":
            #     chr = int(chr.replace("chr", ""))
            # else:
            #     chr = int(chr)
            self.resid_ldr = null_model.resid_ldr - loco_preds.data_reader(chr)
        else:
            self.resid_ldr = null_model.resid_ldr

        if voxels is None:
            self.voxels = np.arange(self.bases.shape[0])
        else:
            self.voxels = voxels

        # self.vset_ld, self.block_size = self._get_sparse_ld_matrix()
        self.vset_ld = self._get_band_ld_matrix()
        
    def _get_band_ld_matrix(self):
        """
        Creating a banded sparse LD matrix with the bandwidth being
        the length of the largest gene
        
        """
        bandwidth = max([len(x) for x in self.numeric_idx_list])
        diagonal_data = list()
        banded_data = list()
        banded_row = list()
        banded_col = list()

        for start in range(0, self.n_variants, bandwidth):
            end1 = start + bandwidth
            end2 = end1 + bandwidth
            vset_block1 = self.vset[start:end1].astype(np.uint16)
            vset_block2 = self.vset[start:end2].astype(np.uint16)
            ld_rec = vset_block1 @ vset_block2.T
            ld_rec_row, ld_rec_col = ld_rec.nonzero()
            ld_rec_row += start
            ld_rec_col += start
            ld_rec_data = ld_rec.data

            diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
            mask = (np.abs(ld_rec_row - ld_rec_col) <= bandwidth) & (
                ld_rec_col > ld_rec_row
            )
            banded_row.append(ld_rec_row[mask])
            banded_col.append(ld_rec_col[mask])
            banded_data.append(ld_rec_data[mask])

        diagonal_data = np.concatenate(diagonal_data)
        banded_row = np.concatenate(banded_row)
        banded_col = np.concatenate(banded_col)
        banded_data = np.concatenate(banded_data)
        shape = np.array([self.n_variants, self.n_variants])

        lower_row = banded_col
        lower_col = banded_row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([banded_row, lower_row, diag_row_col])
        full_col = np.concatenate([banded_col, lower_col, diag_row_col])
        full_data = np.concatenate([banded_data, banded_data, diagonal_data])

        vset_ld = csr_matrix((full_data, (full_row, full_col)), shape=shape)
        return vset_ld

    def _get_sparse_ld_matrix(self):
        positions = np.array(self.locus.locus.position.collect())
        ld = SparseBandedLD(self.vset, positions, 3000000, self.threads)
        diag, data, row, col, shape = ld.data
        
        lower_row = col
        lower_col = row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([row, lower_row, diag_row_col])
        full_col = np.concatenate([col, lower_col, diag_row_col])
        full_data = np.concatenate([data, data, diag])

        vset_ld = csr_matrix((full_data, (full_row, full_col)), shape=shape)
        return vset_ld, ld.block_size

    def _compute_sumstats(self):
        """
        A bootstrap sample is generated by v_i*\\xi_{ij} for j = 1...r
        Each time partition new variant sets

        """
        rand_v = np.random.randn(self.n_subs).reshape(-1, 1)
        resid_ldr_rand = self.resid_ldr * rand_v
        inner_ldr = np.dot(resid_ldr_rand.T, resid_ldr_rand).astype(np.float32)
        self.var = np.sum(np.dot(self.bases, inner_ldr) * self.bases, axis=1)
        self.var /= self.n_subs - self.n_covars  # (N, )
        self.half_ldr_score = self.vset @ resid_ldr_rand 
        # self.numeric_idx_list = self._partition_genome()

    def _variant_set_test(self, sample_id):
        """
        A wrapper function of variant set test for multiple sets

        """
        # rv_sumstats = RVsumstats(f"{self.temp_path}_bootstrap")
        # rv_sumstats.calculate_var()
        # numeric_idx_list = self._partition_genome(rv_sumstats.block_size)

        # cluster_size_list = list() 
        # vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
        # for numeric_idx in numeric_idx_list:
        #     half_ldr_score, cov_mat, maf, is_rare = rv_sumstats.parse_data(
        #         numeric_idx
        #     )
        #     if half_ldr_score is None:
        #         continue
        #     vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, None)
        #     pvalues = vset_test.do_inference()
        #     cluster_size = (pvalues["STAAR-O"] < self.sig_thresh).sum() 
        #     cluster_size_list.append(cluster_size)

        sig_pvalues_list = []
    
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [
                executor.submit(self._variant_set_test_, sample_id, gene_id, numeric_idx)
                for gene_id, numeric_idx in enumerate(self.numeric_idx_list)
            ]
            
            for future in futures:
                result = future.result()
                if result is not None:
                    sig_pvalues_list.append(result)
        
        return sig_pvalues_list

    def _variant_set_test_(self, sample_id, gene_id, numeric_idx):
        """
        Testing a single variant set
        
        """
        vset_test = VariantSetTest(self.bases, self.var)
        half_ldr_score, cov_mat, maf, mac = self._parse_data(
            numeric_idx
        )
        if half_ldr_score is None or np.sum(mac) < self.cmac_min:
            return None
        if self.phred_cate is not None:
            annot = self.phred_cate[numeric_idx]
        else:
            annot = None
        is_rare = mac < self.mac_thresh
        vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, annot)
        pvalues = vset_test.do_inference(self.annot_name)
        # cluster_size = (pvalues["STAAR-O"] < self.sig_thresh).sum()
        pvalues.insert(0, "INDEX", self.voxels+1)
        sig_pvalues = pvalues.loc[pvalues["STAAR-O"] < self.sig_thresh, ["INDEX", "STAAR-O"]]
        if len(sig_pvalues) > 0:
            sig_pvalues.insert(0, "GENE_ID", gene_id+1)
            sig_pvalues.insert(0, "SAMPLE_ID", sample_id+1)
        else:
            sig_pvalues = None

        return sig_pvalues
    
    def _parse_data(self, numeric_idx):
        """
        Extracting data for a variant set to test
        
        """
        # if numeric_idx[-1] - numeric_idx[0] >= self.block_size[numeric_idx[0]]:
        #     return None, None, None, None
        half_ldr_score = np.array(self.half_ldr_score[numeric_idx])
        vset_half_covar_proj = np.array(self.vset_half_covar_proj[numeric_idx])
        vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
        cov_mat = np.array((vset_ld - vset_half_covar_proj @ vset_half_covar_proj.T))
        maf = self.maf[numeric_idx]
        mac = self.mac[numeric_idx]
        
        return half_ldr_score, cov_mat, maf, mac
        
    def _partition_genome(self):
        """
        randomly partition genotype data into intervals with length 2-50
        
        """
        # numeric_idx_list = list()
        # n_variants = len(self.block_size) - 1
        # left = 0
        # while left < n_variants:
        #     if self.block_size[left] > 2:
        #         max_interval_len = min(50, self.block_size[left])
        #         right = left + np.random.choice(list(range(2, max_interval_len)), 1)[0]
        #         numeric_idx_list.append(list(range(left, min(right, n_variants))))
        #         left = right + 2
        #     else:
        #         left += 10
        # mean_length = int(np.mean([len(numeric_idx) for numeric_idx in numeric_idx_list]))
        # self.logger.info(f"{len(numeric_idx_list)} variant sets (mean size {mean_length}) to analyze.")

        # return numeric_idx_list

        numeric_idx_list = list()
        left = 0
        right = 0
        while right < self.n_variants:
            right += np.random.choice(list(range(2, 10)), 1)[0]
            numeric_idx_list.append(list(range(left, min(right, self.n_variants))))
            left = right + 2
            right += 2
            
        return numeric_idx_list

    def cluster_analysis(self, sample_id):
        """
        The main function for computing cluster size for a bootstrap sample

        Parameters:
        ------------
        sample_id: int, bootstrap sample id (0-based)

        """
        self._compute_sumstats()
        sig_pvalues_list = self._variant_set_test(sample_id)

        return sig_pvalues_list


def creating_mask(locus, variant_sets, variant_category, vset, maf, mac):
    """
    Creating masks for a variant category and split into genes

    Parameters:
    ------------
    locus: a hail.Table of locus info
    variant_sets: a pd.DataFrame of genes 
    variant_category: variant category
    vset: (m, n) csr_matrix of genotype
    maf: a np.array of MAF
    mac: a np.array of MAC

    Returns:
    ---------
    chr: chromosome of the genotype
    gene_numeric_idxs: a list of list of variant idxs for each gene
    phred_cate: a np.array of functional annotations
    annot_name: annotation names
    vset: genotype of the extracted variants 
    maf: MAF of the extracted variants
    mac: MAC of the extracted variants
    
    """
    locus = locus.add_index("idx")
    variant_type = locus.variant_type.collect()[0]
    geno_ref = locus.reference_genome.collect()[0]
    mask = Coding(locus, variant_type)
    mask_idx = mask.category_dict[variant_category]
    numeric_idx, phred_cate = mask.parse_annot(mask_idx)
    
    chr = locus.aggregate(hl.agg.take(locus.locus.contig, 1)[0])
    if locus.locus.dtype.reference_genome.name == "GRCh38":
        chr = int(chr.replace("chr", ""))
    else:
        chr = int(chr)

    vset = vset[numeric_idx]
    maf = maf[numeric_idx]
    mac = mac[numeric_idx]

    gene_numeric_idxs = list()
    positions = np.array(locus.locus.position.collect())[numeric_idx]
    for _, gene in variant_sets.iterrows():
        _, start, end = parse_interval(gene[1], geno_ref)
        start_idx = find_loc(positions, start)
        end_idx = find_loc(positions, end) + 1
        if start_idx == -1 or positions[start_idx] != start:
            start_idx += 1
        if end_idx > start_idx + 1 and end_idx - start_idx < 1000: # 2 <= n_variants <= 1000
            gene_numeric_idxs.append(list(range(start_idx, end_idx)))

    return chr, gene_numeric_idxs, phred_cate, mask.annot_name, vset, maf, mac


def check_input(args, log):
    if args.sparse_genotype is None:
        raise ValueError("--sparse-genotype is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")
    if args.variant_category is None:
        raise ValueError("--variant-category is required")
    args.variant_category = args.variant_category.lower()
    if args.variant_category not in {
        "plof",
        "plof_ds",
        "missense",
        "disruptive_missense",
        "synonymous",
        "ptv",
        "ptv_ds",
        }:
        raise ValueError(f"invalid variant category: {args.variant_category}")
    if args.n_bootstrap is None:
        args.n_bootstrap = 50
        log.info("Set #bootstrap as 50")
    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    if args.sig_thresh is None:
        args.sig_thresh = 2.5e-6
        log.info("Set significance threshold as 2.5e-6")
    if args.cmac_min is None:
        args.cmac_min = 2
        log.info(f"Set --cmac-min as default 2")


def run(args, log):
    # checking if input is valid
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # reading data and selecting LDRs
        log.info(f"Read null model from {args.null_model}")
        null_model = NullModel(args.null_model)
        null_model.select_ldrs(args.n_ldrs)
        null_model.select_voxels(args.voxels)

        # reading sparse genotype data
        sparse_genotype = SparseGenotype(args.sparse_genotype)
        log.info(f"Read sparse genotype data from {args.sparse_genotype}")
        log.info(f"{sparse_genotype.vset.shape[1]} subjects and {sparse_genotype.vset.shape[0]} variants.")

        # reading loco preds
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
                sparse_genotype.ids.index,
                null_model.ids,
                loco_preds.ids,
                args.keep,
            )
        else:
            common_ids = ds.get_common_idxs(
                sparse_genotype.ids.index, null_model.ids, args.keep
            )
        common_ids = ds.remove_idxs(common_ids, args.remove)

        # extract and align subjects with the genotype data
        null_model.keep(common_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(common_ids)} common subjects in the data.")
        log.info(
            (
                f"{null_model.covar.shape[1]} fixed effects in the covariates (including the intercept) "
                "after removing redundant effects.\n"
            )
        )

        if args.loco_preds is not None:
            loco_preds.keep(common_ids)
        else:
            loco_preds = None

        # log.info(f"Processing sparse genetic data ...")
        if args.extract_locus is not None:
            args.extract_locus = read_extract_locus(args.extract_locus, args.grch37, log)
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)
        
        sparse_genotype.keep(common_ids)
        sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        sparse_genotype.extract_chr_interval(args.chr_interval)
        sparse_genotype.extract_maf(args.maf_min, args.maf_max)
        sparse_genotype.extract_mac(args.mac_min, args.mac_max)

        # reading annotation
        log.info(f"Read functional annotations from {args.annot_ht}")
        annot = hl.read_table(args.annot_ht)
        sparse_genotype.annotate(annot)
        vset, locus, maf, mac = sparse_genotype.parse_data()

        # creating mask
        (
            chr, gene_numeric_idxs, phred_cate, annot_name, vset, maf, mac
        ) = creating_mask(locus, args.variant_sets, args.variant_category, vset, maf, mac)
        log.info((f"Using {len(gene_numeric_idxs)} genes of "
                  f"{vset.shape[0]} {args.variant_category} variants in wild bootstrap. "
                  "Genes with >1000 variants were excluded."))

        # wild bootstrap
        cluster = RVcluster(
            null_model, 
            vset, 
            chr, 
            gene_numeric_idxs, 
            phred_cate,
            annot_name,
            maf, 
            mac, 
            args.sig_thresh, 
            args.mac_thresh,
            args.cmac_min,
            args.threads, 
            loco_preds,
            args.voxels,
        )

        for i in tqdm(
            range(args.n_bootstrap), desc=f"{args.n_bootstrap} bootstrap samples"
        ):
            log.info(f"Doing bootstrap sample {i+1} ...")
            start_time = time.time()
            sig_pvalues_list = cluster.cluster_analysis(i)
            with open(args.out + ".txt", "a") as file:
                for sig_pvalues in sig_pvalues_list:
                    sig_pvalues = sig_pvalues.to_csv(
                        sep="\t", header=False, na_rep="NA", index=None, float_format="%.5e"
                    )
                # file.write("\n".join(str(x) for x in sig_pvalues_df) + "\n")
                    file.write(sig_pvalues)
            elapsed_time = int((time.time() - start_time))
            log.info(f"done ({elapsed_time}s)")

        # save results
        log.info(f"\nSaved significant associations to {args.out}.txt")

    finally:
        if "loco_preds" in locals() and args.loco_preds is not None:
            loco_preds.close()

        clean(args.out)
