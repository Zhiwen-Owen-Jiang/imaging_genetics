import numpy as np
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


def flip_snps(snps_mt):
    """
    Flipping variants with MAF > 0.5, and creating an annotation for maf
    TODO: make sure the MT has info field

    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype (n, m)
    
    Returns:
    ---------
    snps_mt: a MatrixTable of genotype (n, m)
        
    """
    snps_mt = snps_mt.annotate_entries(
    flipped_n_alt_alleles=hl.if_else(
        snps_mt.info.AF[0] > 0.5,
        2 - snps_mt.GT.n_alt_alleles(),
        snps_mt.GT.n_alt_alleles()
        )
    )   
    snps_mt = snps_mt.annotate_rows(
        maf=hl.if_else(
            snps_mt.info.AF[0] > 0.5,
            1 - snps_mt.info.AF[0],
            snps_mt.info.AF[0]
        )
    ) 
    return snps_mt


def extract_maf(snps_mt, maf_thresh=0.01):
    """
    Extracting variants with a MAF < maf_thresh
    
    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype (n, m)
    maf_thresh: a float number between 0 and 0.5
    
    Returns:
    ---------
    snps_mt: a MatrixTable of genotype (n, m)
    
    """
    if 'maf' not in snps_mt.row:
        raise ValueError('generate a `maf` row before using `filter_maf`')
    snps_mt = snps_mt.filter_rows(snps_mt.maf <= maf_thresh)
    return snps_mt


def annotate_rare_variants(snps_mt, mac_thresh=10):
    """
    Annotating if variants have a MAC < mac_thresh
    
    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype (n, m)
    mac_thresh: a int number greater than 0
    
    Returns:
    ---------
    snps_mt: a MatrixTable of genotype (n, m)
    
    """
    snps_mt = snps_mt.annotate_rows(
        is_rare=hl.if_else(((snps_mt.info.AC < mac_thresh) | 
                    (snps_mt.info.AN - snps_mt.info.AC < mac_thresh)),
                   True, False)
    )
    return snps_mt


def get_genotype_numpy(snps, idx):
    subset_snps = snps.filter_rows(idx)
    n_snps = subset_snps.count_rows()
    collected_data_list = subset_snps.GT.n_alt_alleles().collect()
    subset_snps_numpy = np.array(collected_data_list).reshape(n_snps, -1)
    return subset_snps_numpy
