import hail as hl
from tqdm import tqdm
import numpy as np
from heig.wgs.wgs import RVsumstats
from heig.wgs.vsettest import VariantSetTest 
from heig.wgs.utils import *
from heig.wgs.coding import format_output


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
        self.annot_cols = annot_cols
        if variant_type == 'snv' and annot is not None:
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
        if "annot" in self.rv_sumstats.locus and self.annot_cols is not None:
            if idx is not None:
                filtered_annot = self.rv_sumstats.locus.filter(idx)
            else:
                filtered_annot = self.rv_sumstats.locus
            numeric_idx = filtered_annot.idx.collect()
            annot = filtered_annot.select(*self.annot_cols).collect()
            annot = np.array(
                [
                    [getattr(row, col) for col in self.annot_cols]
                    for row in annot
                ]
            )
        else:
            annot = None
        maf = np.array(filtered_annot.maf.collect())
        is_rare = np.array(filtered_annot.is_rare.collect())

        return numeric_idx, annot, maf, is_rare

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
        numeric_idx, annot, maf, is_rare = self._parse_annot(idx)
        half_ldr_score, cov_mat = self.rv_sumstats.parse_data(numeric_idx)

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
            window_length=None
        ):
        """
        Parameters:
        ------------
        window_length: size of sliding window
        
        """
        super().__init__(rv_sumstats, variant_type, annot, annot_cols)
        self.geno_ref = self.rv_sumstats.locus.reference_genome.collect()[0]
        self.chr = self.rv_sumstats.locus.aggregate(hl.agg.take(self.rv_sumstats.locus.contig, 1)[0])
        self.start = self.rv_sumstats.locus.aggregate(hl.agg.min(self.rv_sumstats.locus))
        self.end = self.rv_sumstats.locus.aggregate(hl.agg.max(self.rv_sumstats.locus))
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

    def parse_window_data(self):
        """
        Parsing data for each window
        
        """
        for start, end in self.windows:
            interval = hl.locus_interval(self.chr, start, end, reference_genome=self.geno_ref)
            window = interval.contains(self.rv_sumstats.locus)
            half_ldr_score, cov_mat, maf, is_rare, annot = self.parse_data(window)

            yield half_ldr_score, cov_mat, maf, is_rare, annot


def vset_analysis(rv_sumstats, variant_type, vset_test, 
                  annot, annot_cols, window_length, log):
    # getting annotations and specific categories of variants
    rv_sumstats.annotate(annot)
    log.info(f"{rv_sumstats.n_variants} variants overlapping in summary statistics and annotations.")

    # analysis
    pvalues = dict()
    if window_length is None:
        general_annot = GeneralAnnotation(rv_sumstats, variant_type, annot, annot_cols)
        half_ldr_score, cov_mat, maf, is_rare, annot = general_annot.parse_data()
        if maf.shape[0] <= 1:
            log.info(f"Less than 2 variants, skip the analysis.")
        else:
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, annot)
            log.info(
                f"Doing analysis the variant set ({vset_test.n_variants} variants) ..."
            )
            pvalues = vset_test.do_inference(general_annot.annot_cols)
            pvalues['vset'] = {
                "n_variants": vset_test.n_variants,
                "pvalues": pvalues,
            }
    else:
        sliding_window = SlidingWindow(rv_sumstats, variant_type, annot, annot_cols, window_length)
        n_windows = len(sliding_window.windows)
        log.info(f"Partitioned the variant set into {n_windows} windows")
        for i, *results in tqdm(enumerate(sliding_window.parse_window_data()), total=n_windows, desc="Analyzing windows"):
            half_ldr_score, cov_mat, maf, is_rare, annot = results
            if maf.shape[0] <= 1:
                log.info(f"Less than 2 variants, skip window {i+1}.")
            else:
                vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, annot)
                log.info(
                    f"Doing analysis for window {i+1} ({vset_test.n_variants} variants) ..."
                )
                pvalues = vset_test.do_inference(sliding_window.annot_cols)
                pvalues[f'window{i+1}'] = {
                    "n_variants": vset_test.n_variants,
                    "pvalues": pvalues,
                }

    return pvalues


def check_input(args, log):
    # required arguments
    if args.rv_sumstats is None:
        raise ValueError("--rv-sumstats is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
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
    if args.annot_ht is not None:
        annot = hl.read_table(args.annot_ht)
    else:
        annot = None

    # single gene analysis
    vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
    pvalues = vset_analysis(
        rv_sumstats, 
        rv_sumstats.variant_type, 
        vset_test,
        annot, 
        args.annot_cols, 
        args.window_length, 
        log
    )

    # format output
    for window_idx, results in pvalues.items():
        results = format_output(
            pvalues["pvalues"],
            pvalues["n_variants"],
            rv_sumstats.voxel_idxs,
            window_idx
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