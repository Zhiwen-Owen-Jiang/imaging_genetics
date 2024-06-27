import numpy as np


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


Annotation_catalog_name = dict()
for k, v in Annotation_name_catalog.items():
    Annotation_catalog_name[v] = k


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


def extract_variant_type(snps, variant_type):
    """
    Extracting variants with specified type

    Parameters:
    ------------
    snps: a MatrixTable of annotated vcf
    variant_type: one of ('variant', 'snv', 'indel')
    
    Returns:
    ---------
    snps: a MatrixTable of annotated vcf
    
    """
    snps = snps.filter_rows(snps.variant_type == variant_type)
    return snps


def extract_gene(start, end, snps, gene_name=None):
    """
    Extacting a gene with the gene name
    snps should have a position column 

    Parameters:
    ------------
    start: start position
    end: end position
    snps: a MatrixTable of annotated vcf
    gene_name: gene name, if specified, start and end will be ignored
    
    Returns:
    ---------
    snps: a MatrixTable of annotated vcf

    """
    if gene_name is None:
        snps = snps.filter_rows((snps.position >= start) & (snps.position <= end))
    else:
        gencode_info = snps.fa[Annotation_name_catalog['GENCODE.Info']]
        snps = snps.filter_rows(gene_name in gencode_info)
    return snps


def fillna_flip_snps(snps):
    """
    Filling NAs in genotypes as 0, and flipping those with MAF > 0.5

    Parameters:
    ------------
    snps: a numpy.array of genotype (n, m)
    
    Returns:
    ---------
    snps: a numpy.array of genotype (n, m)
        
    """
    snps = np.nan_to_num(snps)
    maf = np.mean(snps, axis=0) // 2
    snps[:, maf > 0.5] = 2 - snps[:, maf > 0.5]
    return snps


def get_genotype_numpy(snps, idx):
    subset_snps = snps.filter_rows(idx)
    n_snps = subset_snps.count_rows()
    collected_data_list = subset_snps.GT.n_alt_alleles().collect()
    subset_snps_numpy = np.array(collected_data_list).reshape(n_snps, -1)
    return subset_snps_numpy
