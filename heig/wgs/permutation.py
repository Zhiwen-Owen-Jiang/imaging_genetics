import logging
import h5py
import numpy as np
import hail as hl
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix
from scipy.stats import chi2
import heig.input.dataset as ds
from heig.wgs.null import NullModel
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.mt import SparseGenotype
from heig.wgs.coding import Coding
from heig.utils import find_loc
from heig.wgs.utils import *

"""
Using permutation to test variant sets with a cMAC < 100

Input
1. images
2. annot-ht
3. variant-sets
4. sparse-genotype
5. n-permutes

Output
1. null distribution of test statistics

Steps
1. determine a list of independent voxels

"""

class Permutation:
    """
    Computing null distribution of burden/skat test
    for variant sets with small cMAC

    """

    def __init__(
            self, 
            mask,
            threads=1,
        ):
        """
        Parameters:
        ------------
        mask: an instance of CreatingMask
        threads: number of threads

        """
        self.vset_dict = mask.vset_dict
        self.cov_mat_dict = mask.cov_mat_dict
        self.resid_voxels = mask.resid_voxels
        self.var = mask.var
        self.n_masks = mask.n_masks
        self.n_points = {k: mask.resid_voxels.shape[1]*v for k,v in self.n_masks.items()}
        self.total_points = 5*10**7
        self.cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                          (10,14), (15,20), (21,30), (31,60), (61,100)]
        self.threads = threads
        self.n_subs = self.resid_voxels.shape[0]
        self.sig_thresh = chi2.ppf(1 - 2.5e-6, 1)
        self.logger = logging.getLogger(__name__)

        self.burden_stats_denom_dict = self._get_burden_stats_denom()

    def _get_burden_stats_denom(self):
        burden_stats_denom_dict = {
            (2,2): list(), (3,3): list(), (4,4): list(), (5,5): list(), 
            (6,7): list(), (8,9): list(), (10,14): list(), (15,20): list(), 
            (21,30): list(), (31,60): list(), (61,100): list()
        }
        for bin, cov_mat_list in self.cov_mat_dict.items():
            for cov_mat in cov_mat_list:
                burden_stats_denom_dict[bin].append(self.var * np.sum(cov_mat))
        return burden_stats_denom_dict

    def _permute(self):
        resid_voxels_rand = self.resid_voxels[np.random.permutation(self.n_subs)]
        return resid_voxels_rand

    def _variant_set_test(self, resid_voxels_rand, bin):
        """
        A wrapper function of variant set test for multiple sets

        """
        burden_stats_list = []
    
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [
                executor.submit(
                    self._variant_set_test_, bin, idx, resid_voxels_rand
                )
                for idx in range(self.n_masks[bin])
            ]
            
            for future in futures:
                result = future.result()
                burden_stats_list.append(result)
        
        burden_stats = np.concatenate(burden_stats_list)
        return burden_stats

    def _variant_set_test_(self, bin, idx, resid_voxels_rand):
        """
        Testing a single variant set
        
        """
        vset = self.vset_dict[bin][idx]
        burden_stats_denom = self.burden_stats_denom_dict[bin][idx]
        half_score = vset @ resid_voxels_rand
        burden_stats = np.sum(half_score, axis=0) ** 2 / burden_stats_denom

        return burden_stats
        
    def run(self):
        """
        The main function for permutation

        """
        burden_sig_stats_dict = {
            (2,2): list(), (3,3): list(), (4,4): list(), (5,5): list(), 
            (6,7): list(), (8,9): list(), (10,14): list(), (15,20): list(), 
            (21,30): list(), (31,60): list(), (61,100): list()
        }
        burden_count_dict = {
            (2,2): 0, (3,3): 0, (4,4): 0, (5,5): 0, 
            (6,7): 0, (8,9): 0, (10,14): 0, (15,20): 0, 
            (21,30): 0, (31,60): 0, (61,100): 0
        }
        finished_bins = set()
        while len(finished_bins) < 11:
            resid_voxels_rand = self._permute()
            for bin in self.cmac_bins:
                if bin not in finished_bins:
                    burden_stats = self._variant_set_test(resid_voxels_rand, bin)
                    burden_stats = burden_stats[burden_stats > self.sig_thresh]
                    burden_sig_stats_dict[bin].append(burden_stats)
                    burden_count_dict[bin] += self.n_points[bin] 
                    if burden_count_dict[bin] > self.total_points:
                        finished_bins.add(bin)
                        self.logger.info(f"cMAC bin {bin} finished.")

        for bin, sig_stats in burden_sig_stats_dict.items():
            burden_sig_stats_dict[bin] = np.sort(np.concatenate(sig_stats))

        return burden_sig_stats_dict, burden_count_dict
    

class CreatingMask:
    """
    Generating summary statistics for each variant set
    
    """
    cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                 (10,14), (15,20), (21,30), (31,60), (61,100)]
    def __init__(
            self,
            null_model,
            locus, 
            variant_sets, 
            variant_category, 
            vset,
            maf, 
            mac, 
            cmac_min, 
            cmac_max,
            loco_preds,
    ):
        self.locus = locus
        self.variant_sets = variant_sets
        self.variant_category = variant_category
        self.vset = vset
        self.resid_ldr = null_model.resid_ldr
        self.covar = null_model.covar
        self.n_subs, self.n_covars = self.covar.shape
        self.bases = null_model.bases
        self.maf = maf
        self.mac = mac
        
        self.geno_ref, self.chr, self.positions = self._extract_variant_category()
        self.resid_voxels, self.var = self._misc(loco_preds)
        self.gene_numeric_idxs = self._parse_genes(cmac_min, cmac_max)
        self.n_masks = {k:len(v) for k, v in self.gene_numeric_idxs.items()}
        self.vset_ld = self._get_band_ld_matrix()
        self.vset_dict, self.cov_mat_dict = self._sumstats()

    def _extract_variant_category(self):
        self.locus = self.locus.add_index("idx")
        variant_type = self.locus.variant_type.collect()[0]
        geno_ref = self.locus.reference_genome.collect()[0]
        chr = self.locus.aggregate(hl.agg.take(self.locus.locus.contig, 1)[0])
        if geno_ref == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)
        
        mask = Coding(self.locus, variant_type)
        mask_idx = mask.category_dict[self.variant_category]
        numeric_idx, _ = mask.parse_annot(mask_idx, False)

        self.vset = self.vset[numeric_idx]
        self.maf = self.maf[numeric_idx]
        self.mac = self.mac[numeric_idx]
        positions = np.array(self.locus.locus.position.collect())[numeric_idx]

        return geno_ref, chr, positions
    
    def _misc(self, loco_preds):
        if loco_preds is not None:
            self.resid_ldr = self.resid_ldr - loco_preds.data_reader(self.chr)
        resid_voxels = np.dot(self.resid_ldr, self.bases.T)
        inner_ldr = np.dot(self.resid_ldr.T, self.resid_ldr).astype(np.float32)
        var = np.sum(np.dot(self.bases, inner_ldr) * self.bases, axis=1)
        var /= self.n_subs - self.n_covars  # (N, )
        
        return resid_voxels, var
    
    def _parse_genes(self, cmac_min, cmac_max):
        gene_numeric_idxs = {
            (2,2): list(), (3,3): list(), (4,4): list(), (5,5): list(), 
            (6,7): list(), (8,9): list(), (10,14): list(), (15,20): list(), 
            (21,30): list(), (31,60): list(), (61,100): list()
        }
        all_bins = list(gene_numeric_idxs.keys())
        for _, gene in self.variant_sets.iterrows():
            _, start, end = parse_interval(gene[1], self.geno_ref)
            start_idx = find_loc(self.positions, start)
            end_idx = find_loc(self.positions, end) + 1
            if start_idx == -1 or self.positions[start_idx] != start:
                start_idx += 1
            if end_idx > start_idx + 1:
                numeric_idxs = list(range(start_idx, end_idx))
                cmac = np.sum(self.mac[numeric_idxs])
                if cmac >= 2 and cmac <= 100:
                    bin_idx = find_loc([2,3,4,5,6,8,10,15,21,31,61], cmac)
                    gene_numeric_idxs[all_bins[bin_idx]].append(numeric_idxs)
                    # gene_numeric_idxs[gene[0]] = numeric_idxs   
                         
        for bin, numeric_idx_list in gene_numeric_idxs.items():
            if len(numeric_idx_list) == 0:
                raise ValueError(f"no genes for cMAC bin {bin}")

        return gene_numeric_idxs
    
    def _get_band_ld_matrix(self):
        """
        Creating a banded sparse LD matrix with the bandwidth being
        the length of the largest gene
        
        """
        bandwidth = 100
        n_variants = len(self.maf)
        diagonal_data = list()
        banded_data = list()
        banded_row = list()
        banded_col = list()

        for start in range(0, n_variants, bandwidth):
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
        shape = np.array([n_variants, n_variants])

        lower_row = banded_col
        lower_col = banded_row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([banded_row, lower_row, diag_row_col])
        full_col = np.concatenate([banded_col, lower_col, diag_row_col])
        full_data = np.concatenate([banded_data, banded_data, diagonal_data])

        vset_ld = csr_matrix((full_data, (full_row, full_col)), shape=shape)
        return vset_ld
    
    def _sumstats(self):
        covar_U, _, covar_Vt = np.linalg.svd(self.covar, full_matrices=False)
        half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        vset_half_covar_proj = self.vset @ half_covar_proj

        vset_dict = {
            (2,2): list(), (3,3): list(), (4,4): list(), (5,5): list(), 
            (6,7): list(), (8,9): list(), (10,14): list(), (15,20): list(), 
            (21,30): list(), (31,60): list(), (61,100): list()
        }
        cov_mat_dict = {
            (2,2): list(), (3,3): list(), (4,4): list(), (5,5): list(), 
            (6,7): list(), (8,9): list(), (10,14): list(), (15,20): list(), 
            (21,30): list(), (31,60): list(), (61,100): list()
        }
        for bin, numeric_idx_list in self.gene_numeric_idxs.items():
            for numeric_idx in numeric_idx_list:
                x = np.array(vset_half_covar_proj[numeric_idx])
                vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
                vset_dict[bin].append(self.vset[numeric_idx])
                cov_mat_dict[bin].append(np.array((vset_ld - x @ x.T)))
            
        return vset_dict, cov_mat_dict
            

def select_voxels_greedy(corr, corr_threshold=0.3):
    """
    Select a subset of voxels such that the absolute correlation
    between any two selected voxels is less than corr_threshold.

    Parameters:
    ------------
    corr: (N, N) np.array of voxel correlations
    corr_threshold : correlation threshold to enforce

    Returns:
    ---------
    selected_voxels: a list of voxel indices that survive the correlation filter

    """
    voxel_order = np.arange(corr.shape[0])

    selected_voxels = []
    for voxel in voxel_order:
        too_correlated = False
        for s in selected_voxels:
            if abs(corr[voxel, s]) >= corr_threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected_voxels.append(voxel)

    return selected_voxels


def check_input(args, log):
    if args.voxels is None:
        raise ValueError("--voxels is required")
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
    # if args.cmac_min is None:
    #     raise ValueError("--cmac-min is required")
    # if args.cmac_max is None:
    #     raise ValueError("--cmac-max is required")

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
    # if args.n_bootstrap is None:
    #     args.n_bootstrap = 20000
    #     log.info("Set #permutation as 20000")


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
        log.info((f"{sparse_genotype.vset.shape[1]} subjects and "
                  f"{sparse_genotype.vset.shape[0]} variants."))

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
                f"{null_model.covar.shape[1]} fixed effects in the covariates "
                "(including the intercept) after removing redundant effects.\n"
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
        mask = CreatingMask(
            null_model,
            locus, 
            args.variant_sets, 
            args.variant_category,
            vset,
            maf, 
            mac, 
            args.cmac_min, 
            args.cmac_max,
            loco_preds,
        )

        log.info((f"Using {mask.n_masks} genes "
                  f"{args.variant_category} variants in permutation."
        ))

        # permutation
        permutation = Permutation(mask, args.threads)
        burden_sig_stats_dict, burden_count_dict = permutation.run()
        with h5py.File(f"{args.out}_burden_perm.h5", 'w') as file:
            for bin, sig_stats in burden_sig_stats_dict.items():
                bin_str = "_".join(str(x) for x in bin)
                dataset = file.create_dataset(bin_str, data=sig_stats)
                dataset.attrs["count"] =  burden_count_dict[bin]

        # save results
        log.info(f"\nSaved permutation results to {args.out}_burden_perm.h5")

    finally:
        if "loco_preds" in locals() and args.loco_preds is not None:
            loco_preds.close()

        clean(args.out)