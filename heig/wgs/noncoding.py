import re
import numpy as np
import pandas as pd
import hail as hl



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

# prepared dataframe 
# library(GenomicFeatures)
# txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
# promGobj <- promoters(genes(txdb), upstream = 3000, downstream = 3000)
# promGdf <- data.frame(promGobj)
promGdf = pd.DataFrame() 


class Noncoding:
    def __init__(self, snps, gene_name, *args, **kwargs):
        """
        Parameters:
        ------------
        snps: a hail.MatrixTable of genotype data with annotation attached
        for a specific variant type
        
        """
        self.snps = snps
        self.extract_variants(gene_name, *args, **kwargs)
        self.anno_pred = self.get_annotation()

    def extract_variants(self, gene_name, *args, **kwargs):
        raise NotImplementedError

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
    
    def split_anno(self, annos, split_re, which):
        res = list()
        for anno in annos:
            if which != -1:
                res.append(re.split(split_re, anno)[which])
            else:
                res.extend(re.split(split_re, anno))
        return np.array(res)


class UpDown(Noncoding):
    def __init__(self, snps, gene_name):
        self.gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        super().__init__(snps, gene_name)

    def extract_variants(self, gene_name, type):
        """
        type is 'upstream' or 'downstream'
        
        """
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy()
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(gencode_info, ',', -1), gene_name) != -1
        self.snps = self.snps.filter_rows((is_in) & (self.gencode_category == type))


class UTR(Noncoding):
    def __init__(self, snps, gene_name):
        self.gencode_category = snps.fa[Annotation_name_catalog['GENCODE.Category']]
        super().__init__(snps, gene_name)

    def extract_variants(self, gene_name):
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy()
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(gencode_info, '(', 0), gene_name) != -1
        self.snps = self.snps.filter_rows((is_in) & (self.gencode_category in {'UTR3', 'UTR5', 'UTR5;UTR3'}))


class Promoter(Noncoding):
    def __init__(self, snps, gene_name):
        super().__init__(snps, gene_name)

    def extract_variants(self, gene_name, type):
        cage = self.snps.fa[Annotation_name_catalog[type]] != ''
        gencode_info = self.snps.fa[Annotation_name_catalog['GENCODE.Info']].to_numpy()
        # is_in = np.char.find(gencode_info, gene_name) != -1
        is_in = np.char.find(self.split_anno(gencode_info, '[\(\),;\\-]', 0), gene_name) != -1

        is_prom = np.full(self.snps.shape[0], dtype=bool)
        for _, row in promGdf.iterrows():
            if row['seqnames'] == self.snps.chr:
                start = row['start']
                end = row['end']
                is_prom = is_prom | ((self.snps.position >= start) & (self.snps.position <= end))
        self.snps = self.snps.filter_rows((cage) & (is_in) & (is_prom))


class Enhancer(Noncoding):
    def __init__(self, snps, gene_name):
        super().__init__(snps, gene_name)

    def extract_variants(self, gene_name, type):
        genehancer = self.snps.fa[Annotation_name_catalog['GeneHancer']] != ''
        cage = self.snps.fa[Annotation_name_catalog['DHS']] != ''
        
        genehancer1 = self.split_anno(genehancer, '=', 3)
        genehancer2 = self.split_anno(genehancer1, ';', 0)
        is_in = np.char.find(genehancer2, gene_name) != -1

        self.snps = self.snps.filter_rows((cage) & (genehancer) & (is_in))



