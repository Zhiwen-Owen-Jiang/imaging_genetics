import os
import numpy as np
from abc import ABC, abstractmethod
import hail as hl
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.utils import *


class Noncoding(ABC):
    def __init__(self, annot, variant_type, chr, start, end, type=None):
        """
        Parameters:
        ------------
        annot: a hail.Table of annotations with key ('locus', 'alleles') and hail.struct of annotations
        variant_type: variant type, one of ('variant', 'snv, 'indel')
        chr: chromosome
        start: start position
        end: end position
        type: specific type of non-coding variants

        """
        self.annot = annot
        self.chr, self.start, self.end = chr, start, end
        self.type = type
        self.gencode_category = self.annot.annot[Annotation_name_catalog['GENCODE.Category']]
        self.variant_idx = self.extract_variants()

        if variant_type == "snv":
            self.annot_cols = [
                Annotation_name_catalog[annot_name] for annot_name in Annotation_name
            ]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None
    
    @abstractmethod
    def extract_variants(self):
        """
        Extracting variants by boolean indices

        """
        pass

    def parse_annot(self):
        """
        Parsing annotations and converting to np.array

        Returns:
        ---------
        numeric_idx: a list of numeric indices for extracting sumstats
        phred_cate: a np.array of annotations
        
        """
        filtered_annot = self.annot.filter(self.variant_idx)
        numeric_idx = filtered_annot.idx.collect()
        if len(numeric_idx) <= 1:
            return numeric_idx, None
        if self.annot_cols is not None:
            annot_phred = filtered_annot.annot.select(*self.annot_cols).collect()
            phred_cate = np.array(
                [
                    [getattr(row, col) for col in self.annot_cols]
                    for row in annot_phred
                ]
            )
        else:
            phred_cate = None

        return numeric_idx, phred_cate


class UpDown(Noncoding):
    def extract_variants(self):
        """
        type is 'upstream' or 'downstream'

        """
        variant_idx = self.gencode_category == self.type
        return variant_idx


class UTR(Noncoding):
    def extract_variants(self):
        set1 = hl.literal({'UTR3', 'UTR5', 'UTR5;UTR3'})
        variant_idx = set1.contains(self.gencode_category)
        return variant_idx


class Promoter(Noncoding):
    def extract_variants(self):
        """
        type is 'CAGE' or 'DHS'

        """
        cage = hl.is_defined(self.annot.annot[Annotation_name_catalog[self.type]])

        geno_ref = self.annot.reference_genome.collect()[0]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        main_dir = os.path.dirname(os.path.dirname(base_dir))
        promGdf = hl.import_table(
            os.path.join(main_dir, f'misc/wgs/promGdf_{geno_ref}.csv'),
            no_header=False,
            delimiter=',',
            impute=True
        )
        
        promGdf = promGdf.filter(promGdf['seqnames'] == self.chr)
        promGdf = promGdf.filter(promGdf['start'] >= self.start)
        promGdf = promGdf.filter(promGdf['end'] <= self.end)

        intervals = promGdf.aggregate(
            hl.agg.collect(
                hl.interval(
                    start=hl.locus(promGdf['seqnames'], promGdf['start'], reference_genome=geno_ref),
                    end=hl.locus(promGdf['seqnames'], promGdf['end'], reference_genome=geno_ref),
                    includes_end=False
                )
            )
        )
        
        if len(intervals) > 0:
            is_prom = hl.any(lambda interval: interval.contains(self.annot.locus), intervals)
        else:
            is_prom = hl.literal({'foo'}).contains(self.gencode_category) # create all False indices
        variant_idx = cage & is_prom

        return variant_idx


class Enhancer(Noncoding):
    def extract_variants(self):
        """
        type is 'CAGE' or 'DHS'

        """
        genehancer = self.annot.annot[Annotation_name_catalog['GeneHancer']]
        is_genehancer = hl.is_defined(genehancer)
        cage = hl.is_defined(self.annot.annot[Annotation_name_catalog[self.type.upper()]])
        variant_idx = cage & is_genehancer
        return variant_idx


def noncoding_vset_analysis(rv_sumstats, annot, variant_type, vset_test, variant_category, log):
    """
    Single noncoding variant set analysis

    Parameters:
    ------------
    rv_sumstats: a RVsumstats instance
    annot: a hail.Table of locus containing annotations
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
        one of ('all', 'upstream', 'downstream', 'promoter_cage', 'promoter_dhs',
        'enhancer_cage', 'enhancer_dhs')
    log: a logger

    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # extracting specific variant category
    category_class_map = {
        'upstream': (UpDown, 'upstream'),
        'downstream': (UpDown, 'downstream'),
        'utr': (UTR, None),
        'promoter_cage': (Promoter, 'CAGE'),
        'promoter_dhs': (Promoter, 'DHS'),
        'enhancer_cage': (Enhancer, 'CAGE'),
        'enhancer_dhs': (Enhancer, 'DHS')
    }

    # extracting variants in sumstats and annot
    rv_sumstats.annotate(annot)
    chr, start, end = rv_sumstats.get_interval()

    # individual analysis
    cate_pvalues = dict()
    for category, (_category_class_, type) in category_class_map.items():
        if variant_category[0] == 'all' or category in variant_category:
            category_class = _category_class_(rv_sumstats.locus, variant_type, chr, start, end, type)
            numeric_idx, phred_cate = category_class.parse_annot()
            if len(numeric_idx) <= 1:
                log.info(f"Less than 2 variants for {category}, skip.")
                continue
            half_ldr_score, cov_mat, maf, is_rare = rv_sumstats.parse_data(numeric_idx)
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, phred_cate)
            log.info(
                f"Doing analysis for {category} ({vset_test.n_variants} variants) ..."
            )
            pvalues = vset_test.do_inference(category_class.annot_name)
            cate_pvalues[category] = {
                "n_variants": vset_test.n_variants,
                "pvalues": pvalues,
                "chr": chr,
                "start": start,
                "end": end
            }

    return cate_pvalues


def check_input(args, log):
    # required arguments
    if args.rv_sumstats is None:
        raise ValueError("--rv-sumstats is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")

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
            }:
                log.info(f"Ingore invalid variant category {category}.")
            else:
                variant_category.append(category)
    if len(variant_category) == 0:
        raise ValueError("no valid variant category provided")
    
    # if args.maf_max is None:
    #     if args.maf_min is not None and args.maf_min < 0.01 or args.maf_min is None:
    #         args.maf_max = 0.01
    #         log.info(f"Set --maf-max as default 0.01")

    return variant_category


def run(args, log):
    # checking if input is valid
    variant_category = check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting voxels and LDRs
    log.info(f"Read rare variant summary statistics from {args.rv_sumstats}")
    rv_sumstats = RVsumstats(args.rv_sumstats)
    rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
    rv_sumstats.extract_chr_interval(args.chr_interval)
    rv_sumstats.extract_maf(args.maf_min, args.maf_max)
    rv_sumstats.select_ldrs(args.n_ldrs)
    rv_sumstats.select_voxels(args.voxels)
    rv_sumstats.calculate_var()

    # reading annotation
    annot = hl.read_table(args.annot_ht)

    # single gene analysis
    vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var)
    cate_pvalues = noncoding_vset_analysis(
        rv_sumstats,
        annot,
        rv_sumstats.variant_type,
        vset_test,
        variant_category,
        log,
    )

    for cate, cate_results in cate_pvalues.items():
        cate_output = format_output(
            cate_results["pvalues"],
            cate_results["n_variants"],
            rv_sumstats.voxel_idxs,
            cate_results["chr"],
            cate_results["start"],
            cate_results["end"],
            cate,
        )
        out_path = f"{args.out}_{cate}.txt"
        cate_output.to_csv(
            out_path,
            sep="\t",
            header=True,
            na_rep="NA",
            index=None,
            float_format="%.5e",
        )
        log.info(f"\nSave results for {cate} to {out_path}")