import hail as hl
from tqdm import tqdm
import numpy as np
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.vsettest import VariantSetTest 
from heig.wgs.utils import *


"""
TODO: load all data into memory

"""


class GeneralAnnotation:
    """
    Rare variant analysis using general annotations
    
    """

    def __init__(self, rv_sumstats, annot=None, annot_cols=None):
        """
        Parameters:
        ------------
        rv_sumstats: a RVsumstats instance
        annot: a hail.Table of annotations
        annot_cols: annotations used as weights
        
        """
        self.rv_sumstats = rv_sumstats
        self.annot_cols = annot_cols
        if annot is not None:
            self.rv_sumstats.annotate(annot)
        
    def _parse_annot(self, idx):
        """
        Parsing annotations, maf, and is_rare from locus

        Parameters:
        ------------
        idx: a hail.expr of boolean indices to extract variants

        Returns:
        ---------
        numeric_idx: a list of numeric indices for extracting sumstats
        annot: a np.array of annotations
        maf: a np.array of MAF
        is_rare: a np.array of boolean indices indicating MAC < mac_threshold

        """
        if idx is not None:
            filtered_annot = self.rv_sumstats.locus.filter(idx)
        else:
            filtered_annot = self.rv_sumstats.locus
        numeric_idx = filtered_annot.idx.collect()
        if len(numeric_idx) <= 1:
            return numeric_idx, None
        
        if "annot" in self.rv_sumstats.locus.row and self.annot_cols is not None:
            annot = filtered_annot.annot.select(*self.annot_cols).collect()
            annot = np.array(
                [
                    [getattr(row, col) for col in self.annot_cols]
                    for row in annot
                ]
            )
            if annot.dtype == object:
                raise TypeError('annotations must be numerical data')
            if np.isnan(annot).any():
                raise ValueError('missing values are not allowed in annotations')
        else:
            annot = None

        return numeric_idx, annot

    def parse_data(self, idx):
        """
        Parse data for analysis

        Parameters:
        ------------
        idx: a hail.expr of boolean indices to extract variants

        Returns:
        ---------
        half_ldr_score: Z'(I-M)\Xi
        cov_mat: Z'(I-M)Z
        maf: a np.array of MAF
        is_rare: a np.array of boolean indices indicating MAC < mac_threshold
        annot : a np.array of annotations
        
        """
        numeric_idx, annot = self._parse_annot(idx)
        if len(numeric_idx) <= 1:
            half_ldr_score, cov_mat, maf, is_rare = None, None, None, None
        else:
            half_ldr_score, cov_mat, maf, is_rare = self.rv_sumstats.parse_data(numeric_idx)

        return half_ldr_score, cov_mat, maf, is_rare, annot


class SlidingWindow(GeneralAnnotation):
    """
    Rare variant analysis using fixed-length sliding window
    
    """

    def __init__(
            self, 
            rv_sumstats, 
            window_length,
            sliding_length=None,
            annot=None, 
            annot_cols=None, 
        ):
        """
        Parameters:
        ------------
        window_length: size of sliding window (bp)
        sliding_length: step size of moving forward (bp)
        
        """
        super().__init__(rv_sumstats, annot, annot_cols)
        self.geno_ref = self.rv_sumstats.locus.reference_genome.collect()[0]
        self.chr, self.start, self.end = rv_sumstats.get_interval()
        self.window_length = window_length
        if sliding_length is None:
            self.sliding_length = self.window_length // 2
        else:
            self.sliding_length = sliding_length

        self.windows = self._partition_windows()

    def _partition_windows(self):
        """
        Partitioning variant set into overlapping windows
        sliding length is half of window length

        """
        windows = list()
        cur_left = self.start
        cur_right = self.start + self.sliding_length
        while cur_right < self.end:
            windows.append(tuple([cur_left, cur_right]))
            cur_left, cur_right = cur_right, cur_right + self.sliding_length
        windows.append(tuple([cur_left, self.end]))

        return windows

    def parse_window_data(self):
        """
        Parsing data for each window
        
        """
        for start, end in self.windows:
            interval = hl.locus_interval(self.chr, start, end, reference_genome=self.geno_ref)
            window = interval.contains(self.rv_sumstats.locus.locus)
            half_ldr_score, cov_mat, maf, is_rare, annot = self.parse_data(window)

            yield half_ldr_score, cov_mat, maf, is_rare, annot, self.chr, start, end


def vset_analysis(rv_sumstats, vset_test, annot, annot_cols, window_length, log):
    """
    Variant set analysis

    Parameters:
    ------------
    rv_sumstats: a RVsumstats instance
    annot: a hail.Table of locus containing annotations
    vset_test: an instance of VariantSetTest
    annot_cols: a list of columns of annotations to use
    window_length: window length
    log: a logger

    Returns:
    ---------
    pvalues: a dict (keys: vset/window_idx, values: p-value)
    
    """
    # analysis
    all_pvalues = dict()
    if window_length is None:
        general_annot = GeneralAnnotation(rv_sumstats, annot, annot_cols)
        half_ldr_score, cov_mat, maf, is_rare, annot = general_annot.parse_data()
        if half_ldr_score is None:
            log.info(f"Less than 2 variants, skip the analysis.")
        else:
            chr, start, end = rv_sumstats.get_interval()
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, annot)
            log.info(
                f"Doing analysis for variant set ({vset_test.n_variants} variants) ..."
            )
            pvalues = vset_test.do_inference(general_annot.annot_cols)
            all_pvalues['vset'] = {
                "n_variants": vset_test.n_variants,
                "pvalues": pvalues,
                "chr": chr,
                "start": start,
                "end": end
            }
    else:
        sliding_window = SlidingWindow(rv_sumstats, annot, annot_cols, window_length)
        n_windows = len(sliding_window.windows)
        log.info(f"Partitioned the variant set into {n_windows} windows")
        for i, *results in tqdm(enumerate(sliding_window.parse_window_data()), total=n_windows, desc="Analyzing windows"):
            half_ldr_score, cov_mat, maf, is_rare, annot, chr, start, end = results[0]
            if half_ldr_score is None:
                log.info(f"Less than 2 variants, skip window {i+1}.")
            else:
                vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, annot)
                log.info(
                    f"Doing analysis for window {i+1} ({vset_test.n_variants} variants) ..."
                )
                pvalues = vset_test.do_inference(sliding_window.annot_cols)
                all_pvalues[f'window{i+1}'] = {
                    "n_variants": vset_test.n_variants,
                    "pvalues": pvalues,
                    "chr": chr,
                    "start": start,
                    "end": end
                }

    return all_pvalues


def check_input(args, log):
    # required arguments
    if args.rv_sumstats is None:
        raise ValueError("--rv-sumstats is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_cols is not None:
        args.annot_cols = args.annot_cols.split(",")
    if args.window_length is not None and args.window_length < 2:
        raise ValueError('--window-length must be greater than 2') 
    # if args.maf_max is None:
    #     if args.maf_min is not None and args.maf_min < 0.01 or args.maf_min is None:
    #         args.maf_max = 0.01
    #         log.info(f"Set --maf-max as default 0.01")


def run(args, log):
    # checking if input is valid
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # reading data and selecting voxels and LDRs
        log.info(f"Read rare variant summary statistics from {args.rv_sumstats}")
        rv_sumstats = RVsumstats(args.rv_sumstats)
        rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        rv_sumstats.extract_chr_interval(args.chr_interval)
        # rv_sumstats.extract_maf(args.maf_min, args.maf_max)
        rv_sumstats.select_ldrs(args.n_ldrs)
        rv_sumstats.select_voxels(args.voxels)
        rv_sumstats.calculate_var()

        # reading annotation
        if args.annot_ht is not None:
            annot = hl.read_table(args.annot_ht)
        else:
            annot = None

        # single gene analysis
        vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
        all_pvalues = vset_analysis(
            rv_sumstats, 
            vset_test,
            annot, 
            args.annot_cols, 
            args.window_length, 
            log
        )

        # format output
        for window_idx, window_results in all_pvalues.items():
            results = format_output(
                window_results["pvalues"],
                window_results["n_variants"],
                rv_sumstats.voxel_idxs,
                window_results["chr"],
                window_results["start"],
                window_results["end"],
                window_idx
            )
            out_path = f"{args.out}_{window_idx}.txt"
            results.to_csv(
                out_path,
                sep="\t",
                header=True,
                na_rep="NA",
                index=None,
                float_format="%.5e",
            )
            log.info(f"Save {window_idx} results to {out_path}")
    finally:
        clean(args.out)