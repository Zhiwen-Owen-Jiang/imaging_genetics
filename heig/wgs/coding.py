import hail as hl
import h5py
import pandas as pd
import numpy as np
from heig.wgs.staar import VariantSetTest, cauchy_combination
from heig.wgs.utils import *

"""
Can do multiple genes in parallel

1. Covariates, ldrs, and genotypes need to merge and align
2. Assume varints have been filtered (SNV, indel)
3. the gene has been selected
4. NAs in gene has been filled and every varint is in minor allele

"""


class Coding:
    def __init__(self, snps, variant_type):
        """
        Extracting coding variants, generate annotation, and get index for each category

        Parameters:
        ------------
        snps: a hail.MatrixTable of genotype data with annotation attached
        for a specific gene and variant type
        variant_type: one of ('variant', 'snv', 'indel')
        
        """
        gencode_exonic_category = snps.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        lof_in_coding_snps = ((gencode_exonic_category in {'stopgain', 'stoploss', 'nonsynonymous SNV', 'synonymous SNV'}) |
                              (gencode_category in {'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'}))
        self.snps = snps.filter_rows(lof_in_coding_snps)
        self.gencode_exonic_category = snps.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        self.gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        self.metasvm_pred = snps.fa[Annotation_name_catalog['MetaSVM']]
        self.anno_pred = self.get_annotation()
        self.category_dict = self.get_category(variant_type)

    def get_annotation(self):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        """
        anno_cols = [Annotation_name_catalog[anno_name]
                     for anno_name in Annotation_name]

        # anno_phred = self.snps.fa[anno_cols].to_pandas()
        # anno_phred['cadd_phred'] = anno_phred['cadd_phred'].fillna(0)
        # anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
        # anno_phred['apc_local_nucleotide_diversity2'] = anno_local_div
        
        anno_phred = self.snps.fa.select(*anno_cols)
        anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
        anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
        anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div)    
        return anno_phred

    def get_category(self, variant_type):
        category_dict = dict()
        category_dict['plof'] = (((self.gencode_exonic_category in {'stopgain', 'stoploss'}) |
                                 (self.gencode_category in {'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'})))
        category_dict['synonymous'] = self.gencode_exonic_category == 'synonymous SNV'
        category_dict['missense'] = self.gencode_exonic_category == 'nonsynonymous SNV'
        category_dict['disruptive_missense'] = (
            self.gencode_exonic_category == 'nonsynonymous SNV') & (self.metasvm_pred == 'D')
        category_dict['plof_ds'] = category_dict['plof'] | category_dict['disruptive_missense']

        ptv_snv = ((self.gencode_exonic_category in {'stopgain', 'stoploss'}) | 
                      (self.gencode_category in {'splicing', 'exonic;splicing'}) )
        ptv_indel = (self.gencode_exonic_category in {'frameshift deletion', 'frameshift insertion'})
        if variant_type == 'snv':
            category_dict['ptv'] = ptv_snv
            category_dict['ptv_ds'] = ptv_snv | category_dict['disruptive_missense']
        elif variant_type == 'indel':
            category_dict['ptv'] = ptv_indel
            category_dict['ptv_ds'] = ptv_indel | category_dict['disruptive_missense']
        else:
            category_dict['ptv'] = ptv_snv | ptv_indel 
            category_dict['ptv_ds'] = ptv_snv | ptv_indel | category_dict['disruptive_missense']

        return category_dict


def format_output(res):
    """
    organize pvalues to a structured format

    """
    pass


def single_gene_analysis(snps, start, end, variant_type, vset_test):
    """
    Single gene analysis

    Parameters:
    ------------
    snps: a MatrixTable of annotated vcf
    start: start position of gene
    end: end position of gene
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # extracting specific variant type and the gene
    snps = extract_variant_type(snps, variant_type)
    snps = extract_gene(start, end, snps)

    # individual analysis
    cate_pvalues = dict()
    coding = Coding(snps, variant_type)
    for cate, idx in coding.category_dict.items():
        phred = coding.anno_phred[idx].to_numpy()
        vset = flip_snps(get_genotype_numpy(snps, idx))
        vset_test.input_vset(vset, phred)
        pvalues = vset_test.do_inference()
        cate_pvalues[cate] = pvalues
    cate_pvalues['missense'] = process_missense(cate_pvalues['missense'], 
                                                cate_pvalues['disruptive_missense'])
    
    return cate_pvalues


def process_missense(m_pvalues, dm_pvalues):
    """
    Doing Cauchy combination for missense
    
    """
    cauchy_combination()


def check_input(args, log):
    pass


def run(args, log):
    # checking if input is valid
    start, end = check_input(args, log)

    # reading data
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        var = file['var'][:]
        ids = file['ids'][:]

    bases = np.load(args.bases)
    inner_ldr = np.load(args.inner_ldr)

    vset_test = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    snps = hl.read_matrix_table(args.avcfmt)

    # extracting common ids
    snps_ids = set(snps.s.collect())
    common_ids = snps_ids.intersection(ids)
    snps = snps.filter_cols(hl.literal(common_ids).contains(snps.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis (do parallel)
    res = single_gene_analysis(snps, start, end, vset_test)

   