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


def extract_gene(start, end, snps_mt, gene_name=None):
    """
    Extacting a gene with the gene name
    snps_mt should have a position column 

    Parameters:
    ------------
    start: start position
    end: end position
    snps_mt: a MatrixTable of annotated vcf
    gene_name: gene name, if specified, start and end will be ignored
    
    Returns:
    ---------
    snps_mt: a MatrixTable of annotated vcf

    """
    if gene_name is None:
        snps_mt = snps_mt.filter_rows((snps_mt.locus.position >= start) & (snps_mt.locus.position <= end))
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
    if 'info' not in snps_mt.row:
        snps_mt = hl.variant_qc(snps_mt, name='info')
    snps_mt = snps_mt.annotate_rows(
        is_rare=hl.if_else(((snps_mt.info.AC[-1] < mac_thresh) | 
                    (snps_mt.info.AN - snps_mt.info.AC[-1] < mac_thresh)),
                   True, False)
    )
    return snps_mt


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