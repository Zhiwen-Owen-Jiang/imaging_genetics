import os
import numpy as np
from abc import ABC, abstractmethod
import hail as hl
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.utils import *


class Noncoding(ABC):
    def __init__(self, annot, variant_type, promG_intervals=None):
        """
        Parameters:
        ------------
        annot: a hail.Table of annotations with key ('locus', 'alleles') and hail.struct of annotations
        variant_type: variant type, one of ('variant', 'snv, 'indel')
        promG_intervals: locus intervals of promG

        """
        self.annot = annot
        self.promG_intervals = promG_intervals
        self.gencode_category = self.annot.annot[
            Annotation_name_catalog["GENCODE.Category"]
        ]
        self.genecode_info = self.annot.annot[
            Annotation_name_catalog["GENCODE.Info"]
        ]

        if variant_type == "snv":
            self.annot_cols = [
                Annotation_name_catalog[annot_name] for annot_name in Annotation_name
            ]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None

    @abstractmethod
    def extract_variants(self, gene, type=None):
        """
        Extracting variants by boolean indices

        Parameters:
        ------------
        gene: gene_name
        type: specific type of non-coding variants

        """
        pass

    def parse_annot(self, variant_idx):
        """
        Parsing annotations and converting to np.array

        Returns:
        ---------
        numeric_idx: a list of numeric indices for extracting sumstats
        phred_cate: a np.array of annotations

        """
        filtered_annot = self.annot.filter(variant_idx)
        numeric_idx = filtered_annot.idx.collect()
        all_locus = filtered_annot.locus.collect()
        chr = all_locus[0].contig
        start = all_locus[0].position
        end = all_locus[-1].position

        if len(numeric_idx) <= 1:
            return numeric_idx, None
        if self.annot_cols is not None:
            annot_phred = filtered_annot.annot.select(*self.annot_cols).collect()
            phred_cate = np.array(
                [[getattr(row, col) for col in self.annot_cols] for row in annot_phred]
            )
        else:
            phred_cate = None

        return numeric_idx, phred_cate, chr, start, end


class UpDown(Noncoding):
    def extract_variants(self, gene, type):
        """
        type is 'upstream' or 'downstream'

        """
        variant_idx1 = self.gencode_category == type
        variant_idx2 = self.genecode_info.contains(gene)

        return variant_idx1 & variant_idx2


class UTR(Noncoding):
    def extract_variants(self, gene):
        set1 = hl.literal({"UTR3", "UTR5", "UTR5;UTR3"})
        variant_idx1 = set1.contains(self.gencode_category)
        # self.annot = self.annot.annonate(utr=hl.str.split(self.genecode_info, r'\(')[0])
        # variant_idx2 = self.annot.utr == self.gene
        variant_idx2 = self.genecode_info.startswith(gene)

        return variant_idx1 & variant_idx2


class Promoter(Noncoding):
    def __init__(self, annot, variant_type, promG_intervals):
        super().__init__(annot, variant_type, promG_intervals)
        self.annot = self.annot.annotate(split_genes=self.genecode_info.split(r'[(),;\\-]')[0])

    def extract_variants(self, gene, type):
        """
        type is 'CAGE' or 'DHS'

        """
        cage = hl.is_defined(self.annot.annot[Annotation_name_catalog[type]])
        is_prom = hl.any(
            lambda interval: interval.contains(self.annot.locus), self.promG_intervals
        )
        gene_idx = self.annot.split_genes == gene
        variant_idx = cage & is_prom & gene_idx

        return variant_idx


class Enhancer(Noncoding):
    def __init__(self, annot, variant_type, promG_intervals):
        super().__init__(annot, variant_type, promG_intervals)
        genehancer = self.annot.annot[Annotation_name_catalog["GeneHancer"]]
        self.annot = self.annot.annotate(genehancer=genehancer.split('connected_gene=')[1].split(';')[0])

    def extract_variants(self, gene, type):
        """
        type is 'CAGE' or 'DHS'

        """
        genehancer = self.annot.annot[Annotation_name_catalog["GeneHancer"]]
        is_genehancer = hl.is_defined(genehancer)
        cage = hl.is_defined(
            self.annot.annot[Annotation_name_catalog[type]]
        )
        gene_idx = self.annot.genehancer == gene
        variant_idx = cage & is_genehancer & gene_idx

        return variant_idx


class NcRNA(Noncoding):
    def extract_variants(self, gene):
        set1 = hl.literal({"ncRNA_exonic", "ncRNA_exonic;splicing", "ncRNA_splicing"})
        variant_idx1 = set1.contains(self.gencode_category)
        # self.annot = self.annot.annonate(ncrna=hl.str.split(self.genecode_info, r';')[0])
        # variant_idx2 = self.annot.ncrna == self.gene
        variant_idx2 = self.genecode_info.contains(gene)

        return variant_idx1 & variant_idx2


def read_promG(geno_ref):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(os.path.dirname(base_dir))
    promGdf = hl.import_table(
        os.path.join(main_dir, f"misc/wgs/promGdf_{geno_ref}.csv"),
        no_header=False,
        delimiter=",",
        impute=True,
    )

    intervals = promGdf.aggregate(
        hl.agg.collect(
            hl.interval(
                start=hl.locus(
                    promGdf["seqnames"], promGdf["start"], reference_genome=geno_ref
                ),
                end=hl.locus(
                    promGdf["seqnames"], promGdf["end"], reference_genome=geno_ref
                ),
                includes_end=False,
            )
        )
    )

    return intervals

def noncoding_vset_analysis(
    rv_sumstats, annot, variant_sets, variant_type, vset_test, variant_category, log
):
    """
    Single noncoding variant set analysis

    Parameters:
    ------------
    rv_sumstats: a RVsumstats instance
    annot: a hail.Table of locus containing annotations
    variant_sets: a pd.DataFrame of multiple variant sets to test
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
        one of ('all', 'upstream', 'downstream', 'promoter_cage', 'promoter_dhs',
        'enhancer_cage', 'enhancer_dhs', 'ncrna')
    log: a logger

    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)

    """
    # extracting specific variant category
    category_class_map = {
        "upstream": (UpDown, "upstream"),
        "downstream": (UpDown, "downstream"),
        "utr": (UTR, None),
        "promoter_cage": (Promoter, "CAGE"),
        "promoter_dhs": (Promoter, "DHS"),
        "enhancer_cage": (Enhancer, "CAGE"),
        "enhancer_dhs": (Enhancer, "DHS"),
        "ncrna": (NcRNA, None)
    }

    # extracting variants in sumstats and annot
    annot_locus = rv_sumstats.annotate(annot)

    # creating annotation class
    category_class_dict = dict()
    promGdf_intervals = None
    for category, (_category_class_, type) in category_class_map.items():
        if variant_category[0] == "all" or category in variant_category:
            if category == "promoter_cage" or category == "promoter_dhs" and promGdf_intervals is None:
                promGdf_intervals = read_promG(rv_sumstats.geno_ref)

            category_class = _category_class_(
                annot_locus, variant_type, promGdf_intervals
            )
            category_class_dict[category] = category_class, type

    for _, gene in variant_sets.iterrows():
        # individual analysis
        cate_pvalues = dict()

        for category, (category_class, type) in category_class_dict.items():
            variant_idx = category_class.extract_variants(gene[0], type)
            numeric_idx, phred_cate, chr, start, end = category_class.parse_annot(variant_idx)
            if len(numeric_idx) <= 1:
                log.info(f"Less than 2 variants for {category}, skip.")
                continue
            half_ldr_score, cov_mat, maf, is_rare = rv_sumstats.parse_data(numeric_idx)
            if half_ldr_score is None:
                continue
            if np.sum(maf * rv_sumstats.n_subs * 2) < 10:
                log.info(f"Skip {category} (< 10 total MAC).")
                continue
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, phred_cate)
            log.info(
                f"Doing analysis for {category} ({vset_test.n_variants} variants) ..."
            )
            pvalues = vset_test.do_inference(category_class.annot_name)
            cate_pvalues[category] = {
                "n_variants": vset_test.n_variants,
                "pvalues": pvalues,
            }

    yield gene[0], cate_pvalues, chr, start, end


def check_input(args, log):
    # required arguments
    if args.rv_sumstats is None:
        raise ValueError("--rv-sumstats is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")
    if args.variant_sets is None:
        raise ValueError("--variant-sets is required")
    log.info(f"{args.variant_sets.shape[0]} genes in --variant-sets.")

    if args.staar_only:
        log.info("Saving STAAR-O results only.")
    if args.sig_thresh is not None:
        log.info(f"Saving results with a p-value less than {args.sig_thresh}")

    if args.variant_category is None:
        variant_category = ["all"]
        log.info(f"Set --variant-category as default 'all'.")
    else:
        variant_category = list()
        args.variant_category = [x.lower() for x in args.variant_category.split(",")]
        for category in args.variant_category:
            if category == "all":
                variant_category = ["all"]
                break
            elif category not in {
                "upstream",
                "downstream",
                "utr",
                "promoter_cage",
                "promoter_dhs",
                "enhancer_cage",
                "enhancer_dhs",
                "ncrna"
            }:
                log.info(f"Ingore invalid variant category {category}.")
            else:
                variant_category.append(category)
    if len(variant_category) == 0:
        raise ValueError("no valid variant category provided")

    return variant_category


def run(args, log):
    # checking if input is valid
    variant_category = check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # reading data and selecting voxels and LDRs
        log.info(f"Read rare variant summary statistics from {args.rv_sumstats}")
        rv_sumstats = RVsumstats(args.rv_sumstats)
        rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        rv_sumstats.extract_chr_interval(args.chr_interval)
        rv_sumstats.select_ldrs(args.n_ldrs)
        rv_sumstats.select_voxels(args.voxels)
        rv_sumstats.calculate_var()

        # reading annotation
        annot = hl.read_table(args.annot_ht)

        # single gene analysis
        vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
        all_vset_test_pvalues = noncoding_vset_analysis(
            rv_sumstats,
            annot,
            args.variant_sets,
            rv_sumstats.variant_type,
            vset_test,
            variant_category,
            log,
        )
        
        index_file = IndexFile(f"{args.out}_result_index.txt")
        log.info(f"Write result index file to {args.out}_result_index.txt")

        for set_name, chr, start, end, cate_pvalues in all_vset_test_pvalues.items():
            cate_output = format_output(
                cate_pvalues,
                rv_sumstats.voxel_idxs,
                args.staar_only,
                args.sig_thresh
            )
            if cate_output is not None:
                out_path = f"{args.out}_{set_name}.txt"
                index_file.write_index(
                    set_name,
                    chr,
                    start,
                    end,
                    out_path
                )
                cate_output.to_csv(
                    out_path,
                    sep="\t",
                    header=True,
                    na_rep="NA",
                    index=None,
                    float_format="%.5e",
                )
                log.info(
                    f"Save results for {set_name} to {args.out}_{set_name}.txt"
                )
            else:
                log.info(f"No significant results for {set_name}.")
    finally:
        clean(args.out)
