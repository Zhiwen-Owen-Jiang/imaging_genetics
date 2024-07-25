import re
import h5py
import numpy as np
import pandas as pd
import hail as hl
from heig.wgs.staar import VariantSetTest
from heig.wgs.utils import *


# prepared dataframe
# library(GenomicFeatures)
# txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
# promGobj <- promoters(genes(txdb), upstream = 3000, downstream = 3000)
# promGdf <- data.frame(promGobj)
promGdf = pd.DataFrame()

# this should be an abstract class
class Noncoding:
    def __init__(self, snps_mt, gene_name, variant_type, type=None):
        """
        Parameters:
        ------------
        snps_mt: a hail.MatrixTable of genotype data with annotation attached
        for a specific variant type
        gene_name: gene name
        variant_type: variant type, one of ('variant', 'snv, 'indel')

        """
        self.snps_mt = snps_mt
        self.variant_type = variant_type
        self.gene_name = gene_name
        self.type = type
        self.variant_idx = self.extract_variants()
        # self.annot_pred = self.get_annotation(variant_idx)
        # self.vset = self.fillna_flip_snps(self.snps_mt)

        self.gencode_category = self.snps_mt.fa[Annotation_name_catalog['GENCODE.Category']]
        self.gencode_info = self.snps_mt.fa[Annotation_name_catalog['GENCODE.Info']]

    def extract_variants(self):
        raise NotImplementedError


class UpDown(Noncoding):
    def extract_variants(self):
        """
        type is 'upstream' or 'downstream'

        """
        is_in = self.gencode_info.split(',').contains(self.gene_name)
        variant_idx = is_in & (self.gencode_category == self.type)
        return variant_idx


class UTR(Noncoding):
    def extract_variants(self):
        is_in = self.gencode_info.split('(')[0] == self.gene_name # check it
        set1 = hl.literal({'UTR3', 'UTR5', 'UTR5;UTR3'})
        variant_idx = is_in & set1.contains(self.gencode_category)
        return variant_idx


class Promoter(Noncoding):
    def extract_variants(self):
        """
        type is 'cage' or 'dhs'

        """
        cage = hl.is_defined(self.snps_mt.fa[Annotation_name_catalog[self.type]])
        is_in = self.gencode_info.split('[\(\),;\\-]')[0] == self.gene_name

        # this should be updated
        is_prom = np.full(self.snps_mt.shape[0], dtype=bool)
        for _, row in promGdf.iterrows():
            if row['seqnames'] == self.snps_mt.locus.chr:
                start = row['start']
                end = row['end']
                is_prom = is_prom | ((self.snps_mt.locus.position >= start) & (self.snps_mt.locus.position <= end))
        # self.snps_mt = self.snps_mt.filter_rows((cage) & (is_in) & (is_prom))
        variant_idx = (cage) & (is_in) & (is_prom)
        return variant_idx


class Enhancer(Noncoding):
    def extract_variants(self):
        """
        type is 'cage' or 'dhs'

        """
        genehancer = self.snps_mt.fa[Annotation_name_catalog['GeneHancer']]
        is_genehancer = hl.is_defined(genehancer)
        cage = hl.is_defined(self.snps_mt.fa[Annotation_name_catalog[self.type]])
        genehancer1 = genehancer.split('=')[3]
        genehancer2 = genehancer1.split(';')[0]
        is_in = genehancer2 == self.gene_name
        variant_idx = cage & is_genehancer & is_in
        return variant_idx


def single_gene_analysis(snps_mt, gene_name, category, variant_type, vset_test):
    # extracting specific variant type
    category_class_map = {
        'upstream': (UpDown, 'upstream'),
        'downstream': (UpDown, 'downstream'),
        'utr': (UTR, None),
        'promoter_cage': (Promoter, 'CAGE'),
        'promoter_dhs': (Promoter, 'DHS'),
        'enhancer_cage': (Enhancer, 'CAGE'),
        'enhancer_dhs': (Enhancer, 'DHS')
    }

    # individual analysis
    if category == 'all':
        pvalues = dict()
        for cate, (cate_class, type) in category_class_map.items():
            pvalues[cate] = single_category_analysis(vset_test, cate_class, snps_mt,
                                                     gene_name, variant_type, type)
    else:
        cate_class, type = category_class_map[category]
        pvalues = single_category_analysis(vset_test, cate_class, snps_mt,
                                           gene_name, variant_type, type)

    return pvalues


def single_category_analysis(vset_test, cate_class, snps_mt, gene_name, variant_type, type):
    cate = cate_class(snps_mt, gene_name, variant_type, type)
    phred = cate.annot_pred.to_numpy()
    vset_test.input_vset(cate.vset, phred)
    pvalues = vset_test.do_inference()

    return pvalues


def check_input(args, log):
    pass


def run(args, log):
    # checking if input is valid
    check_input(args, log)

    # reading data
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        var = file['var'][:]
        ids = file['ids'][:]

    bases = np.load(args.bases)
    inner_ldr = np.load(args.inner_ldr)

    vset_test = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    snps_mt = hl.read_matrix_table(args.avcfmt)

    # extracting common ids
    snps_mt_ids = set(snps_mt.s.collect())
    common_ids = snps_mt_ids.intersection(ids)
    snps_mt = snps_mt.filter_cols(hl.literal(common_ids).contains(snps_mt.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis (do parallel)
    res = single_gene_analysis(
        snps_mt, args.gene_name, args.category, args.variant_type, vset_test)
