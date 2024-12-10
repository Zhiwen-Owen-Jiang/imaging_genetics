import hail as hl
import numpy as np
import pandas as pd
from functools import reduce
from heig.wgs.wgs import RVsumstats
from heig.wgs.vsettest import VariantSetTest, cauchy_combination
from heig.wgs.utils import *
from heig.wgs.coding import format_output


"""
1. Rare variant analysis for general annotations
2. Sliding window w/ or w/o annotations

TODO:
1. general annotations may be transformed as FAVOR

"""

class GeneralAnnotation:
    """
    Rare variant analysis using general annotations
    
    """

    def __init__(self, rv_sumstats, variant_type, annot=None, annot_cols=None):
        """
        Parameters:
        ------------
        rv_sumstats: a RVsumstats instance
        variant_type: one of ('variant', 'snv', 'indel')
        annot: a hail.Table of annotations
        annot_cols: annotations used as weights
        
        """
        self.rv_sumstats = rv_sumstats
        if variant_type == 'snv' and annot is not None:
            if annot_cols is not None:
                annot = annot.select(*annot_cols)
            self.rv_sumstats = self.rv_sumstats.annotate(annot)

    def _parse_annot(self, idx):
        """
        Parsing annotations

        Parameters:
        ------------
        idx: 

        """
        if self.annot is not None and self.annot_cols is not None:
            if idx is not None:
                filtered_annot = self.annot.filter(idx)
            else:
                filtered_annot = self.annot
            annot = filtered_annot.select(*self.annot_cols).collect()
            annot = np.array(
                [
                    [getattr(row, col) for col in self.annot_cols]
                    for row in annot
                ]
            )
        else:
            annot = None

        return annot

    def parse_data(self):
        numeric_idx = self.rv_sumstats.locus.idx.collect()
        half_ldr_score, cov_mat, maf, is_rare = self.rv_sumstats.parse_data(numeric_idx)
        annot = self._parse_annot(numeric_idx)

        return half_ldr_score, cov_mat, maf, is_rare, annot


class SlidingWindow(GeneralAnnotation):
    """
    Rare variant analysis using fixed-length sliding window
    
    """

    def __init__(
            self, 
            rv_sumstats, 
            variant_type, 
            annot=None, 
            annot_cols=None, 
            chr_interval=None, 
            window_length=None
        ):
        """
        Parameters:
        ------------
        chr_interval: chr interval
        window_length: size of sliding window
        
        """
        super().__init__(rv_sumstats, variant_type, annot, annot_cols)
        geno_ref = self.rv_sumstats.locus.reference_genome.collect()[0]
        self.chr, self.start, self.end = parse_interval(chr_interval, geno_ref)
        self.window_length = window_length
        self.windows = self._partition_windows()

    def _partition_windows(self):
        """
        Partitioning variant set into overlapping windows
        sliding length is half of window length

        """
        windows = list()
        sliding_length = self.window_length // 2
        cur_left = self.start
        cur_right = self.start + sliding_length
        while cur_right < self.end:
            windows.append(tuple([cur_left, cur_right]))
            cur_left, cur_right = cur_right, cur_right + sliding_length
        windows.append(tuple([cur_left, self.end]))

        return windows

    def parse_data(self):
        for start, end in self.windows:
            interval = hl.locus_interval(self.chr, start, end, reference_genome=self.geno_ref)
            window = self.rv_sumstats.locus.filter(interval.contains(self.rv_sumstats.locus))
            numeric_idx = window.idx.collect()
            half_ldr_score, cov_mat, maf, is_rare = self.rv_sumstats.parse_data(numeric_idx)
            annot = self._parse_annot(numeric_idx)
            yield half_ldr_score, cov_mat, maf, is_rare, annot


def vset_analysis(rv_sumstats, variant_type, vset_test, 
                  annot, annot_cols, chr_interval, window_length, log):
    # getting annotations and specific categories of variants
    if window_length is None:
        general_annot = GeneralAnnotation(rv_sumstats, variant_type, annot, annot_cols)
        half_ldr_score, cov_mat, maf, is_rare = general_annot.parse_data()
        if maf.shape[0] <= 1:
            log.info(f"Less than 2 variants, skip.")
        else:
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, phred_cate)

    annot = annot.semi_join(rv_sumstats.locus)
    rv_sumstats.semi_join(annot)
    log.info(f"{rv_sumstats.n_variants} variants overlapping in summary statistics and annotations.")


def check_input(args, log):
    # required arguments
    if args.rv_sumstats is None:
        raise ValueError("--rv-sumstats is required")
    if args.maf_max is None and args.maf_min < 0.01:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")

def run(args, log):
    # checking if input is valid
    check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting voxels and LDRs
    log.info(f"Read rare variant summary statistics from {args.rv_sumstats}")
    rv_sumstats = RVsumstats(args.rv_sumstats)
    rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
    rv_sumstats.extract_chr_interval(args.chr_interval)
    rv_sumstats.extract_maf(args.max_min, args.maf_max)
    rv_sumstats.select_ldrs(args.n_ldrs)
    rv_sumstats.select_voxels(args.voxels)

    # reading annotation
    annot = hl.read_table(args.annot_ht)

    # single gene analysis
    vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
    pvalues = vset_analysis(
        rv_sumstats,
        annot,
        rv_sumstats.variant_type,
        vset_test,
        args.annot_names,
        log,
    )

    # format output
    results = format_output(
        pvalues["pvalues"],
        pvalues["n_variants"],
        rv_sumstats.voxel_idxs,
        args.variant_category
    )
    out_path = f"{args.out}.txt"
    results.to_csv(
        out_path,
        sep="\t",
        header=True,
        na_rep="NA",
        index=None,
        float_format="%.5e",
    )
    log.info(f"\nSave results to {out_path}")