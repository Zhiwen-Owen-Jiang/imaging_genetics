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


class Noncoding:
    def __init__(self, snps, gene_name, variant_type, *args, **kwargs):
        """
        Parameters:
        ------------
        snps: a hail.MatrixTable of genotype data with annotation attached
        for a specific variant type
        gene_name: gene name
        variant_type: variant type, one of ('variant', 'snv, 'indel')

        """
        self.snps = snps
        self.variant_type = variant_type
        self.gene_name = gene_name
        variant_idx = self.extract_variants(*args, **kwargs)
        self.anno_pred = self.get_annotation(variant_idx)
        self.vset = self.fillna_flip_snps(self.snps)

    def extract_variants(self, *args, **kwargs):
        raise NotImplementedError

    def get_annotation(self, variant_idx):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        Parameters:
        ------------
        variant_idx: boolean indices for variants in the gene

        Returns:
        ---------
        anno_phred: a MatrixTable of processed annotation for the gene

        """
        if self.variant_type != 'snv':
            anno_phred = self.snps.filter_rows(variant_idx).fa.annotate(null_weight=1)
        else:
            anno_cols = [Annotation_name_catalog[anno_name]
                         for anno_name in Annotation_name]
            anno_phred = self.snps.filter_rows(
                variant_idx).fa.select(*anno_cols)
            anno_phred = anno_phred.annotate(
                cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
            anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
            anno_phred = anno_phred.annotate(
                apc_local_nucleotide_diversity2=anno_local_div)
        return anno_phred

    def split_anno(self, annos, split_re, which):
        """
        Splitting annotation using regular expression

        Parameters:
        ------------
        annos: a np.array of annotation (m, )
        split_re: regular expression of the pattern
        which: which element to keep

        Returns:
        ---------
        res: a np.array of annotation (m, )
        
        """
        res = list()
        for anno in annos:
            if which != -1:
                res.append(re.split(split_re, anno)[which])
            else:
                res.extend(re.split(split_re, anno))
        res = np.array(res)
        return res

    def fillna_flip_snps(self, snps):
        """
        Filling NAs in genotypes as 0, and flipping those with MAF > 0.5

        Parameters:
        ------------
        snps: a hail.MatrixTable of genotype data with annotation attached

        Returns:
        ---------
        snps: a numpy.array of genotype (n, m)

        """
        snps = snps.GT.to_numpy()
        snps = np.nan_to_num(snps)
        maf = np.mean(snps, axis=0) // 2
        snps[:, maf > 0.5] = 2 - snps[:, maf > 0.5]
        return snps


class UpDown(Noncoding):
    def __init__(self, snps, gene_name, variant_type, type):
        self.gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        super().__init__(snps, gene_name, variant_type, type)

    def extract_variants(self, type):
        """
        type is 'upstream' or 'downstream'

        """
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy(
        )
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(gencode_info, ',', -1), self.gene_name) != -1
        variant_idx = (is_in) & (self.gencode_category == type)
        # self.snps = self.snps.filter_rows((is_in) & (self.gencode_category == type))
        return variant_idx


class UTR(Noncoding):
    def __init__(self, snps, gene_name, variant_type):
        self.gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        super().__init__(snps, gene_name, variant_type)

    def extract_variants(self, type=None):
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy(
        )
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(
            gencode_info, '(', 0), self.gene_name) != -1
        variant_idx = (is_in) & (self.gencode_category in {
            'UTR3', 'UTR5', 'UTR5;UTR3'})
        # self.snps = self.snps.filter_rows((is_in) & (self.gencode_category in {'UTR3', 'UTR5', 'UTR5;UTR3'}))
        return variant_idx


class Promoter(Noncoding):
    def __init__(self, snps, gene_name, variant_type, type):
        super().__init__(snps, gene_name, variant_type, type)

    def extract_variants(self, type):
        """
        type is 'cage' or 'dhs'

        """
        cage = self.snps.fa[Annotation_name_catalog[type]] != ''
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy(
        )
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(
            gencode_info, '[\(\),;\\-]', 0), self.gene_name) != -1

        is_prom = np.full(self.snps.shape[0], dtype=bool)
        for _, row in promGdf.iterrows():
            if row['seqnames'] == self.snps.chr:
                start = row['start']
                end = row['end']
                is_prom = is_prom | (
                    (self.snps.position >= start) & (self.snps.position <= end))
        # self.snps = self.snps.filter_rows((cage) & (is_in) & (is_prom))
        variant_idx = (cage) & (is_in) & (is_prom)
        return variant_idx


class Enhancer(Noncoding):
    def __init__(self, snps, gene_name, variant_type, type):
        super().__init__(snps, gene_name, variant_type, type)

    def extract_variants(self, type):
        """
        type is 'cage' or 'dhs'

        """
        genehancer = self.snps.fa[Annotation_name_catalog['GeneHancer']] != ''
        cage = self.snps.fa[Annotation_name_catalog[type]] != ''
        genehancer1 = self.split_anno(genehancer, '=', 3)
        genehancer2 = self.split_anno(genehancer1, ';', 0)
        is_in = np.char.find(genehancer2, self.gene_name) != -1
        # self.snps = self.snps.filter_rows((cage) & (genehancer) & (is_in))
        variant_idx = (cage) & (genehancer) & (is_in)
        return variant_idx


def single_gene_analysis(snps, gene_name, category, variant_type, vset_test):
    # extracting specific variant type
    snps = extract_variant_type(snps, variant_type)
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
            pvalues[cate] = single_category_analysis(vset_test, cate_class, snps,
                                                     gene_name, variant_type, type)
    else:
        cate_class, type = category_class_map[category]
        pvalues = single_category_analysis(vset_test, cate_class, snps,
                                           gene_name, variant_type, type)

    return pvalues


def single_category_analysis(vset_test, cate_class, snps, gene_name, variant_type, type):
    cate = cate_class(snps, gene_name, variant_type, type)
    phred = cate.anno_pred.to_numpy()
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
    snps = hl.read_matrix_table(args.avcfmt)

    # extracting common ids
    snps_ids = set(snps.s.collect())
    common_ids = snps_ids.intersection(ids)
    snps = snps.filter_cols(hl.literal(common_ids).contains(snps.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis (do parallel)
    res = single_gene_analysis(
        snps, args.gene_name, args.category, args.variant_type, vset_test)
