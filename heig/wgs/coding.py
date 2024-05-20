import hail as hl
import h5py
import pandas as pd
import numpy as np
from heig.wgs.staar import VariantSetTest


"""
1. Assume covariates, ldrs, and genotypes have been merged and aligned
2. Assume varints have been filtered (SNV, indel)
3. the gene has been selected
4. NAs in gene has been filled and every varint is in minor allele

"""

Annotation_name_catalog = {
    'rs_num': 'rsid',
    'GENCODE.Category': 'genecode_comprehensive_category',
    'GENCODE.Info': 'genecode_comprehensive_info',
    'GENCODE.EXONIC.Category': 'genecode_comprehensive_exonic_category',
    'MetaSVM': 'metasvm_pred',
    'GeneHancer': 'genehancer',
    'CAGE': 'cage_tc',
    'DHS': 'rdhs',
    'CADD': 'cadd_phred',
    'LINSIGHT': 'linsight',
    'FATHMM.XF': 'fathmm_xf',
    'aPC.EpigeneticActive': 'apc_epigenetics_active',
    'aPC.EpigeneticRepressed': 'apc_epigenetics_repressed',
    'aPC.EpigeneticTranscription': 'apc_epigenetics_transcription',
    'aPC.Conservation': 'apc_conservation',
    'aPC.LocalDiversity': 'apc_local_nucleotide_diversity',
    'aPC.Mappability': 'apc_mappability',
    'aPC.TF': 'apc_transcription_factor',
    'aPC.Protein': 'apc_protein_function'
}

Annotation_name = ("CADD",
                   "LINSIGHT",
                   "FATHMM.XF",
                   "aPC.EpigeneticActive",
                   "aPC.EpigeneticRepressed",
                   "aPC.EpigeneticTranscription",
                   "aPC.Conservation",
                   "aPC.LocalDiversity",
                   "aPC.Mappability",
                   "aPC.TF",
                   "aPC.Protein"
                   )


class Coding:
    def __init__(self, snps):
        """
        Extract coding variants, generate annotation, and get index for each category

        Parameters:
        ------------
        snps: a hail.MatrixTable of genotype data with annotation attached
        
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
        self.category_dict = self.get_category()

    def get_annotation(self):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        """
        anno_cols = [Annotation_name_catalog[anno_name]
                     for anno_name in Annotation_name]

        # anno_phred = self.snps.fa[anno_cols].to_pandas()
        # anno_phred['cadd_phred'] = anno_phred['cadd_phred'].fillna(0)
        # anno_local_div = -10 * \
        #     np.log10(
        #         1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
        # anno_phred['apc_local_nucleotide_diversity2'] = anno_local_div
        
        anno_phred = self.snps.fa.select(*anno_cols)
        anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
        anno_local_div = -10 * \
            np.log10(
                1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
        anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div)    
        return anno_phred

    def get_category(self):
        category_dict = dict()
        category_dict['plof'] = (((self.gencode_exonic_category in {'stopgain', 'stoploss'}) |
                                 (self.gencode_category in {'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'})))
        category_dict['synonymous'] = self.gencode_exonic_category == 'synonymous SNV'
        category_dict['missense'] = self.gencode_exonic_category == 'nonsynonymous SNV'
        category_dict['disruptive_missense'] = (
            self.gencode_exonic_category == 'nonsynonymous SNV') & (self.metasvm_pred == 'D')
        category_dict['plof_ds'] = category_dict['plof'] | category_dict['disruptive_missense']

        return category_dict


def filter_gene(gene_name, gene_list, snps):
    """
    snps should have a position column 

    """
    start, end = gene_list.loc[gene_list['hgnc_symbol']
                               == gene_name, ['start_position', 'end_position']]
    snps = snps.filter_rows((snps.position >= start) & (snps.position <= end))
    return snps


def detect_variants(snps, variant_type):
    """
    variant_type is in ('variant', 'snv', 'indel')

    """
    pass


def fillna_flip_snps(snps):
    """
    Fill NAs in genotypes as 0, and flip those with MAF > 0.5

    Parameters:
    ------------
    snps: a numpy.array of genotype (n, m)
        
    """
    snps = np.nan_to_num(snps)
    maf = np.mean(snps, axis=0) // 2
    snps[:, maf > 0.5] = 2 - snps[:, maf > 0.5]

    return snps


def run(args, log):
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldrs = file['resid_ldrs'][:]
        var = file['var'][:]
        ids = file['ids'][:]

    snps = hl.read_matrix_table(args.avcfmt)

    # extract common ids
    snps_ids = set(snps.s.collect())
    common_ids = snps_ids.intersection(ids)

    snps = snps.filter_cols(hl.literal(common_ids).contains(snps.s))
    