import numpy as np
import pandas as pd
from scipy.stats import chi2
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.utils import *


"""
Computing variant-level associations

"""

class SingleVariant:
    def __init__(self, locus, half_score, cov_mat_diag, var, maf, mac, voxel_idxs):
        self.locus = locus
        self.half_score = half_score
        self.cov_mat_diag = cov_mat_diag
        self.var = var
        self.maf = maf
        self.mac = mac
        self.voxel_idxs = voxel_idxs

    def assoc(self, sig_thresh):
        variant_var = self.var / self.cov_mat_diag
        variant_beta = self.half_score / self.cov_mat_diag
        variant_chisq = variant_beta ** 2 / variant_var
        sig_map = variant_chisq > sig_thresh
        sig_voxels = sig_map.any(axis=0)
        sig_variants_df = list()

        for i, voxel_idx in enumerate(self.voxel_idxs):
            if sig_voxels[i]:
                sig_variants = self.locus[sig_map[:, i]].copy()
                sig_variants["MAF"] = self.maf[sig_map[:, i]]
                sig_variants["MAC"] = self.mac[sig_map[:, i]]
                sig_variants["BETA"] = variant_beta[sig_map[:, i], i]
                sig_variants["SE"] = np.sqrt(variant_var[sig_map[:, i], i])
                sig_variants["Z"] = sig_variants["BETA"] / sig_variants["SE"]
                sig_variants["P"] = chi2.sf(variant_chisq[sig_map[:, i], i], 1)
                sig_variants.insert(0, "INDEX", [voxel_idx + 1] * np.sum(sig_map[:, i]))
                sig_variants_df.append(sig_variants)
        
        sig_variants_df = pd.concat(sig_variants_df, axis=0)
        
        return sig_variants_df
    

def parse_sumstats_data(rv_sumstats):
    numeric_idx = rv_sumstats.locus.idx.collect()
    vset_half_covar_proj = np.array(rv_sumstats.vset_half_covar_proj[numeric_idx])
    vset_ld = rv_sumstats.vset_ld[numeric_idx][:, numeric_idx]
    half_ldr_score = np.array(rv_sumstats.half_ldr_score[numeric_idx])

    half_score = np.dot(half_ldr_score, rv_sumstats.bases.T)
    cov_mat_diag = np.array((vset_ld.diagonal() - np.sum(vset_half_covar_proj ** 2, axis=1))).reshape(-1, 1)
    maf = rv_sumstats.maf[numeric_idx]
    mac = rv_sumstats.mac[numeric_idx]

    locus = rv_sumstats.locus.collect()
    locus_df = [(x.locus.contig, x.locus.position, x.alleles[0], x.alleles[1]) for x in locus]
    locus_df = pd.DataFrame.from_records(locus_df, columns=['CHR', 'POS', 'A2', 'A1']) # first is ref
    
    return half_score, cov_mat_diag, rv_sumstats.var, maf, mac, locus_df


def check_input(args, log):
    if args.rv_sumstats_part1 is None:
        raise ValueError("--rv-sumstats-part1 is required")
    if args.rv_sumstats_part2 is None:
        raise ValueError("--rv-sumstats-part2 is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.sig_thresh is None:
        args.sig_thresh = 5e-8
        log.info("Set significance threshold as 5e-8")


def run(args, log):
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        if args.extract_locus is not None:
            args.extract_locus = read_extract_locus(args.extract_locus, args.grch37, log)
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        # reading data and selecting voxels and LDRs
        log.info((f"Read rare variant summary statistics from "
                  f"{args.rv_sumstats_part1} and {args.rv_sumstats_part2}"))
        rv_sumstats = RVsumstats(args.rv_sumstats_part1, args.rv_sumstats_part2)
        rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        rv_sumstats.extract_chr_interval(args.chr_interval)
        rv_sumstats.extract_maf(args.maf_min, args.maf_max)
        rv_sumstats.extract_mac(args.mac_min, args.mac_max)
        rv_sumstats.select_ldrs(args.n_ldrs)
        rv_sumstats.select_voxels(args.voxels)
        rv_sumstats.calculate_var()

        half_score, cov_mat_diag, maf, mac, var, locus_df = parse_sumstats_data(rv_sumstats)
        single_variant = SingleVariant(locus_df, half_score, cov_mat_diag, var, maf, mac, rv_sumstats.voxel_idxs)
        sig_variants_df = single_variant.assoc(args.sig_thresh)
        sig_variants_df.to_csv(f"{args.out}.txt", sep='\t', index=None, float_format="%.5e")
        log.info(f"\nSaved rare variant association results to {args.out}.txt")

    finally:
        clean(args.out)