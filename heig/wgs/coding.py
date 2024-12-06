import hail as hl
import numpy as np
import pandas as pd
from functools import reduce
from heig.wgs.wgs import WGSsumstats
from heig.wgs.vsettest import VariantSetTest, cauchy_combination
from heig.wgs.utils import *


OFFICIAL_NAME = {
    "plof": "predicted loss of function (pLoF) variants",
    "synonymous": "synonymous variants",
    "missense": "missense variants",
    "disruptive_missense": "disruptive missense variants",
    "plof_ds": "pLoF variants with deleterious score",
    "ptv": "protein truncating variants (PTV)",
    "ptv_ds": "PTV with eleterious score",
}


class Coding:
    def __init__(self, annot, variant_type):
        """
        Extracting coding variants, generate annotation, and get index for each category

        Parameters:
        ------------
        annot: a hail.Table of annotations, semi_merged with locus
        variant_type: one of ('variant', 'snv', 'indel')

        """
        self.annot = annot

        gencode_exonic_category = self.annot[
            Annotation_name_catalog["GENCODE.EXONIC.Category"]
        ]
        gencode_category = self.annot[Annotation_name_catalog["GENCODE.Category"]]
        valid_exonic_categories = hl.literal(
            {"stopgain", "stoploss", "nonsynonymous SNV", "synonymous SNV"}
        )
        valid_categories = hl.literal(
            {"splicing", "exonic;splicing", "ncRNA_splicing", "ncRNA_exonic;splicing"}
        )
        lof_in_coding_annot = valid_exonic_categories.contains(
            gencode_exonic_category
        ) | valid_categories.contains(gencode_category)
        self.annot = self.annot.filter(lof_in_coding_annot)
        if self.annot.count() == 0:
            raise ValueError("no variants remaining")

        self.gencode_exonic_category = self.annot[
            Annotation_name_catalog["GENCODE.EXONIC.Category"]
        ]
        self.gencode_category = self.annot[
            Annotation_name_catalog["GENCODE.Category"]
        ]
        self.metasvm_pred = self.annot[Annotation_name_catalog["MetaSVM"]]
        self.category_dict = self.get_category(variant_type)

        if variant_type == "snv":
            self.annot_cols = [
                Annotation_name_catalog[annot_name] for annot_name in Annotation_name
            ]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None

    def get_category(self, variant_type):
        """
        Extracting different categories of variants

        Parameters:
        ------------
        variant_type: one of ('variant', 'snv', 'indel')

        Returns:
        ---------
        category_dict: a dict containing variant indices

        """
        category_dict = dict()
        set1 = hl.literal({"stopgain", "stoploss"})
        set2 = hl.literal(
            {"splicing", "exonic;splicing", "ncRNA_splicing", "ncRNA_exonic;splicing"}
        )
        set3 = hl.literal({"splicing", "exonic;splicing"})
        set4 = hl.literal({"frameshift deletion", "frameshift insertion"})

        category_dict["plof"] = set1.contains(
            self.gencode_exonic_category
        ) | set2.contains(self.gencode_category)
        category_dict["synonymous"] = self.gencode_exonic_category == "synonymous SNV"
        category_dict["missense"] = self.gencode_exonic_category == "nonsynonymous SNV"
        category_dict["disruptive_missense"] = category_dict["missense"] & (
            self.metasvm_pred == "D"
        )
        category_dict["plof_ds"] = (
            category_dict["plof"] | category_dict["disruptive_missense"]
        )

        ptv_snv = set1.contains(self.gencode_exonic_category) | set3.contains(
            self.gencode_category
        )
        ptv_indel = set4.contains(self.gencode_exonic_category)
        if variant_type == "snv":
            category_dict["ptv"] = ptv_snv
            category_dict["ptv_ds"] = ptv_snv | category_dict["disruptive_missense"]
        elif variant_type == "indel":
            category_dict["ptv"] = ptv_indel
            category_dict["ptv_ds"] = ptv_indel | category_dict["disruptive_missense"]
        else:
            category_dict["ptv"] = ptv_snv | ptv_indel
            category_dict["ptv_ds"] = (
                category_dict["ptv"] | category_dict["disruptive_missense"]
            )

        return category_dict


def coding_vset_analysis(
    wgs_sumstats, annot, variant_type, vset_test, variant_category, log
):
    """
    Single coding variant set analysis

    Parameters:
    ------------
    wgs_sumstats: a WGSsumstats instance
    annot: a hail.Table of annotations
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
        one of ('all', 'plof', 'plof_ds', 'missense', 'disruptive_missense',
        'synonymous', 'ptv', 'ptv_ds')
    use_annotation_weights: if using annotation weights
    log: a logger

    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)

    """
    # getting annotations and specific categories of variants
    annot = annot.semi_join(wgs_sumstats.locus)
    wgs_sumstats.semi_join(annot)
    coding = Coding(annot, variant_type)

    # individual analysis
    cate_pvalues = dict()
    for cate, idx in coding.category_dict.items():
        if variant_category[0] != "all" and cate not in variant_category:
            cate_pvalues[cate] = None
        else:
            half_ldr_score, cov_mat, maf, is_rare = wgs_sumstats.parse_data(idx)
            if maf.shape[0] <= 1:
                log.info(f"Less than 2 variants for {OFFICIAL_NAME[cate]}, skip.")
                continue
            if coding.annot_cols is not None:
                annot_phred = annot.select(*coding.annot_cols).collect()
                phred_cate = np.array(
                    [
                        [getattr(row, col) for col in coding.annot_cols]
                        for row in annot_phred
                    ]
                )
            else:
                phred_cate = None
            vset_test.input_vset(half_ldr_score, cov_mat, maf, is_rare, phred_cate)
            log.info(
                f"Doing analysis for {OFFICIAL_NAME[cate]} ({vset_test.n_variants} variants) ..."
            )
            pvalues = vset_test.do_inference(coding.annot_name)
            cate_pvalues[cate] = {
                "n_variants": vset_test.n_variants,
                "pvalues": pvalues,
            }

    if "missense" in cate_pvalues and "disruptive_missense" in cate_pvalues:
        cate_pvalues["missense"] = process_missense(
            cate_pvalues["missense"], cate_pvalues["disruptive_missense"]
        )

    return cate_pvalues


def process_missense(m_pvalues, dm_pvalues):
    """
    Incoporating disruptive missense results into missense

    Parameters:
    ------------
    m_pvalues: pvalues of missense variants
    dm_pvalues: pvalues of disruptive missense variants

    Returns:
    ---------
    m_pvalues: pvalues of missense variants incoporating disruptive missense results

    """

    dm_pvalues = dm_pvalues["pvalues"]
    n_m_variants = m_pvalues["n_variants"]
    m_pvalues = m_pvalues["pvalues"]

    m_pvalues["SKAT(1,25)-Disruptive"] = dm_pvalues["SKAT(1,25)"]
    m_pvalues["SKAT(1,1)-Disruptive"] = dm_pvalues["SKAT(1,1)"]
    m_pvalues["Burden(1,25)-Disruptive"] = dm_pvalues["Burden(1,25)"]
    m_pvalues["Burden(1,1)-Disruptive"] = dm_pvalues["Burden(1,1)"]
    m_pvalues["ACAT-V(1,25)-Disruptive"] = dm_pvalues["ACAT-V(1,25)"]
    m_pvalues["ACAT-V(1,1)-Disruptive"] = dm_pvalues["ACAT-V(1,1)"]

    columns = m_pvalues.columns.values
    skat_1_25 = np.array([column.startswith("SKAT(1,25)") for column in columns])
    skat_1_1 = np.array([column.startswith("SKAT(1,1)") for column in columns])
    burden_1_25 = np.array([column.startswith("Burden(1,25)") for column in columns])
    burden_1_1 = np.array([column.startswith("Burden(1,1)") for column in columns])
    acatv_1_25 = np.array([column.startswith("ACAT-V(1,25)") for column in columns])
    acatv_1_1 = np.array([column.startswith("ACAT-V(1,1)") for column in columns])

    staar_s_1_25 = cauchy_combination(m_pvalues.loc[:, skat_1_25].values.T)
    staar_s_1_1 = cauchy_combination(m_pvalues.loc[:, skat_1_1].values.T)
    staar_b_1_25 = cauchy_combination(m_pvalues.loc[:, burden_1_25].values.T)
    staar_b_1_1 = cauchy_combination(m_pvalues.loc[:, burden_1_1].values.T)
    staar_a_1_25 = cauchy_combination(m_pvalues.loc[:, acatv_1_25].values.T)
    staar_a_1_1 = cauchy_combination(m_pvalues.loc[:, acatv_1_1].values.T)

    m_pvalues["STAAR-S(1,25)"] = staar_s_1_25
    m_pvalues["STAAR-S(1,1)"] = staar_s_1_1
    m_pvalues["STAAR-B(1,25)"] = staar_b_1_25
    m_pvalues["STAAR-B(1,1)"] = staar_b_1_1
    m_pvalues["STAAR-A(1,25)"] = staar_a_1_25
    m_pvalues["STAAR-A(1,1)"] = staar_a_1_1

    all_columns = [skat_1_25, skat_1_1, burden_1_25, burden_1_1, acatv_1_25, acatv_1_1]
    all_columns = reduce(np.logical_or, all_columns)
    all_columns = np.concatenate([all_columns, np.ones(6, dtype=bool)])
    m_pvalues["STAAR-O"] = cauchy_combination(m_pvalues.loc[:, all_columns].values.T)
    m_pvalues = {"n_variants": n_m_variants, "pvalues": m_pvalues}

    return m_pvalues


def format_output(cate_pvalues, chr, start, end, n_variants, voxels, variant_category):
    """
    organizing pvalues to a structured format

    Parameters:
    ------------
    cate_pvalues: a pd.DataFrame of pvalues of the variant category
    chr: chromosome
    start: start position of the gene
    end: end position of the gene
    n_variants: #variants of the category
    voxels: zero-based voxel idxs of the image
    variant_category: which category of variants to analyze,
        one of ('plof', 'plof_ds', 'missense', 'disruptive_missense',
        'synonymous', 'ptv', 'ptv_ds')

    Returns:
    ---------
    output: a pd.DataFrame of pvalues with metadata

    """
    meta_data = pd.DataFrame(
        {
            "INDEX": voxels + 1,
            "VARIANT_CATEGORY": variant_category,
            "CHR": chr,
            "START": start,
            "END": end,
            "N_VARIANT": n_variants,
        }
    )
    output = pd.concat([meta_data, cate_pvalues], axis=1)
    return output


def check_input(args, log):
    # required arguments
    if args.wgs_sumstats is None:
        raise ValueError("--wgs-sumstats is required")

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
            if category not in {
                "all",
                "plof",
                "plof_ds",
                "missense",
                "disruptive_missense",
                "synonymous",
                "ptv",
                "ptv_ds",
            }:
                log.info(f"Ingore invalid variant category {category}.")
            else:
                variant_category.append(category)
        if len(variant_category) == 0:
            raise ValueError("no valid variant category provided")
        if (
            "missense" in variant_category
            and "disruptive_missense" not in variant_category
        ):
            variant_category.append("disruptive_missense")

    if args.maf_max is None and args.maf_min < 0.01:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")

    start_chr, start_pos, end_pos = process_range(args.chr_interval)

    return start_chr, start_pos, end_pos, variant_category


def run(args, log):
    # checking if input is valid
    chr, start, end, variant_category = check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting voxels and LDRs
    log.info(f"Read WGS/WES summary statistics from {args.wgs_sumstats}")
    wgs_sumstats = WGSsumstats(args.wgs_sumstats)
    wgs_sumstats.select_ldrs(args.n_ldrs)
    wgs_sumstats.select_voxels(args.voxels)

    # reading annotation
    annot = hl.read_table(args.annot)

    # single gene analysis
    vset_test = VariantSetTest(wgs_sumstats.bases, wgs_sumstats.var)
    cate_pvalues = coding_vset_analysis(
        wgs_sumstats,
        annot,
        wgs_sumstats.variant_type,
        vset_test,
        variant_category,
        log,
    )

    # format output
    for cate, cate_results in cate_pvalues.items():
        cate_output = format_output(
            cate_results["pvalues"],
            chr,
            start,
            end,
            cate_results["n_variants"],
            wgs_sumstats.voxel_idxs,
            cate,
        )
        out_path = f"{args.out}_chr{chr}_start{start}_end{end}_{cate}.txt"
        cate_output.to_csv(
            out_path,
            sep="\t",
            header=True,
            na_rep="NA",
            index=None,
            float_format="%.5e",
        )
        log.info(f"\nSave results for {OFFICIAL_NAME[cate]} to {out_path}")