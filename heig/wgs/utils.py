import hail as hl
import numpy as np
import pandas as pd


__all__ = ['Annotation_name_catalog', 'Annotation_catalog_name',
           'Annotation_name', 'preprocess_mt', 'keep_ldrs',
           'remove_dependent_columns', 'extract_align_subjects',
           'get_common_ids']

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


Annotation_name = ["CADD",
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
                   ]


def extract_variant_type(snps_mt, variant_type):
    """
    Extracting variants with specified type

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated vcf
    variant_type: one of ('variant', 'snv', 'indel')
    
    Returns:
    ---------
    snps_mt: a MatrixTable of annotated vcf
    
    """
    if variant_type == 'variant':
        return snps_mt
    elif variant_type == 'snv':
        func = hl.is_snp
    elif variant_type == 'indel':
        func = hl.is_indel
    else:
        raise ValueError('variant_type must be snv, indel or variant')
    snps_mt = snps_mt.annotate_rows(target_type=func(snps_mt.alleles[0], snps_mt.alleles[1]))
    snps_mt = snps_mt.filter_rows(snps_mt.target_type)
    return snps_mt


def extract_gene(snps_mt, geno_ref, chr, start, end, gene_name=None):
    """
    Extacting a gene with the gene name
    snps_mt should have a position column 

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated vcf
    geno_ref: reference genome
    chr: target chromosome
    start: start position
    end: end position
    gene_name: gene name, if specified, start and end will be ignored
    
    Returns:
    ---------
    snps_mt: a MatrixTable of annotated vcf

    """
    chr = str(chr)
    if geno_ref == 'GRCh38':
        chr = 'chr' + chr
        
    if gene_name is None:
        snps_mt = snps_mt.filter_rows((snps_mt.locus.contig == chr) & 
                                      (snps_mt.locus.position >= start) & 
                                      (snps_mt.locus.position <= end))
    else:
        gencode_info = snps_mt.fa[Annotation_name_catalog['GENCODE.Info']]
        snps_mt = snps_mt.filter_rows(gene_name in gencode_info)
    return snps_mt


def flip_snps(snps_mt):
    """
    Flipping variants with MAF > 0.5, and creating an annotation for maf

    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype (n, m)
    
    Returns:
    ---------
    snps_mt: a MatrixTable of genotype (n, m)
        
    """
    if 'info' not in snps_mt.row:
        snps_mt = hl.variant_qc(snps_mt, name='info')
    snps_mt = snps_mt.annotate_entries(
    flipped_n_alt_alleles=hl.if_else(
        snps_mt.info.AF[-1] > 0.5,
        2 - snps_mt.GT.n_alt_alleles(),
        snps_mt.GT.n_alt_alleles()
        )
    )   
    snps_mt = snps_mt.annotate_rows(
        maf=hl.if_else(
            snps_mt.info.AF[-1] > 0.5,
            1 - snps_mt.info.AF[-1],
            snps_mt.info.AF[-1]
        )
    ) 
    return snps_mt


def extract_maf(snps_mt, maf_min=None, maf_max=0.01):
    """
    Extracting variants with a MAF < maf_max
    
    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype (n, m)
    maf_max: a float number between 0 and 0.5
    maf_min: a float number between 0 and 0.5, 
             shoule be smaller than maf_max
    
    Returns:
    ---------
    snps_mt: a MatrixTable of genotype (n, m)
    
    """
    if maf_min is None:
        maf_min = 0
    if maf_min >= maf_max:
        raise ValueError('maf_min is greater than maf_max')
    if 'info' not in snps_mt.row:
        snps_mt = hl.variant_qc(snps_mt, name='info')
    if 'maf' not in snps_mt.row:
        snps_mt = snps_mt.annotate_rows(
            maf=hl.if_else(
                snps_mt.info.AF[-1] > 0.5,
                1 - snps_mt.info.AF[-1],
                snps_mt.info.AF[-1]
            )
        )
    snps_mt = snps_mt.filter_rows((snps_mt.maf >= maf_min) & 
                                  (snps_mt.maf <= maf_max))
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
    if 'info' not in snps_mt.row:
        snps_mt = hl.variant_qc(snps_mt, name='info')
    snps_mt = snps_mt.annotate_rows(
        is_rare=hl.if_else(((snps_mt.info.AC[-1] < mac_thresh) | 
                    (snps_mt.info.AN - snps_mt.info.AC[-1] < mac_thresh)),
                   True, False)
    )
    return snps_mt


def extract_snps(snps_mt, keep_snps):
    keep_snps = hl.literal(set(keep_snps['SNP']))
    snps_mt = snps_mt.filter_rows(keep_snps.contains(snps_mt.rsid))
    return snps_mt


def extract_idvs(snps_mt, keep_idvs):
    """
    
    keep_idvs: a set of ids
    
    """
    keep_idvs = hl.literal(keep_idvs)
    snps_mt = snps_mt.filter_cols(keep_idvs.contains(snps_mt.s))
    return snps_mt


def get_common_ids(ids, snps_mt_ids, keep_idvs=None):
    if keep_idvs is not None:
        keep_idvs = keep_idvs.get_level_values('IID').tolist()
        common_ids = set(keep_idvs).intersection(ids)
    else:
        common_ids = set(ids)
    common_ids = common_ids.intersection(snps_mt_ids)
    return common_ids


def preprocess_mt(snps_mt, geno_ref, *args, keep_snps=None, keep_idvs=None,
                  variant_type='snv', maf_min=None, maf_max=0.01,
                  mac_thresh=10, **kwargs):
    if 'filters' in snps_mt.row:
        snps_mt = snps_mt.filter_rows((hl.len(snps_mt.filters) == 0) | hl.is_missing(snps_mt.filters))
    if keep_snps is not None:
        snps_mt = extract_snps(snps_mt, keep_snps)
    if keep_idvs is not None:
        snps_mt = extract_idvs(snps_mt, keep_idvs)
    snps_mt = hl.variant_qc(snps_mt, name='info')
    snps_mt = extract_variant_type(snps_mt, variant_type)
    snps_mt = extract_maf(snps_mt, maf_min, maf_max)
    snps_mt = flip_snps(snps_mt)
    snps_mt = annotate_rare_variants(snps_mt, mac_thresh)
    if args or kwargs:
        snps_mt = extract_gene(snps_mt, geno_ref, *args, **kwargs)

    return snps_mt


def keep_ldrs(n_ldrs, bases, resid_ldr):
    if bases.shape[1] < n_ldrs:
        raise ValueError('the number of bases is less than --n-ldrs')
    if resid_ldr.shape[1] < n_ldrs:
        raise ValueError('LDR residuals are less than --n-ldrs')
    bases = bases[:, :n_ldrs]
    resid_ldr = resid_ldr[:, :n_ldrs]
    return bases, resid_ldr


def remove_dependent_columns(matrix):
    rank = np.linalg.matrix_rank(matrix)
    if rank < matrix.shape[1]:
        _, R = np.linalg.qr(matrix)
        independent_columns = np.where(np.abs(np.diag(R)) > 1e-10)[0]
        matrix = matrix[:, independent_columns]
    return matrix


def extract_align_subjects(current_id, target_id):
    """
    Extracting and aligning subjects for a dataset based on another dataset
    target_id must be the subset of current_id

    Parameters:
    ------------
    current_id: a list or np.array of ids of the current dataset
    target_id: a list or np.array of ids of the another dataset

    Returns:
    ---------
    index: a np.array of indices such that current_id[index] = target_id

    """
    if not set(target_id).issubset(current_id):
        raise ValueError('targettarget_id must be the subset of current_id')
    n_current_id = len(current_id)
    current_id = pd.DataFrame({'id': current_id, 'index': range(n_current_id)})
    target_id = pd.DataFrame({'id': target_id})
    target_id = target_id.merge(current_id, on='id')
    index = np.array(target_id['index'])
    return index


if __name__ == '__main__':
    main='/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis'
    snps_mt = hl.import_plink(bed=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.bed',
                              bim=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.bim',
                              fam=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.fam',
                              reference_genome='GRCh37')
    snps_mt = hl.variant_qc(snps_mt, name='info')
    snps_mt = extract_variant_type(snps_mt, 'indel')
    snps_mt = extract_maf(snps_mt, 0.05)
    snps_mt = flip_snps(snps_mt)
    snps_mt = annotate_rare_variants(snps_mt, 100)
    snps_mt = extract_gene(1000000, 14559856, snps_mt)