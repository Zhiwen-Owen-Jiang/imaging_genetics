import os
import subprocess
import hail as hl
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from heig.utils import find_loc
import heig.input.dataset as ds
from heig.wgs.utils import extract_idvs, extract_snps


def process_snp_info(snp_info):
    """
    Processing SNP info extracted from MatrixTable to more commonly used format

    Parameters:
    ------------
    snp_info: a pd.DataFrame extracted from MatrixTable

    Returns:
    ---------
    snp_df: a pd.DataFrame with basic SNP info
    
    """
    chr_list = snp_info['locus'].apply(lambda x: x.contig).astype(int)
    if len(set(chr_list)) != 1:
        raise ValueError('the VCF file must contain only one chromosome')
    pos_list = snp_info['locus'].apply(lambda x: x.position).astype(int)
    ref_list = snp_info['alleles'].apply(lambda x: x[0])
    alt_list = snp_info['alleles'].apply(lambda x: x[1])

    snp_df = pd.DataFrame({'CHR': chr_list, 'POS': pos_list,
                           'REF': ref_list, 'ALT': alt_list})
    return snp_df


def extract_annotation(xsv, chr, i, out_path, db_path):
    """
    Annotating variants in a block

    Parameters:
    ------------
    xsv: a pd.DataFrame extracted from MatrixTable
    chr: target chromosome
    i: ith block
    out_path: directory saving the variant info of VCF
    db_path: directory saving annotation data from FAVORannotator

    """
    code = f'{xsv} join --left VarInfo '
    code += os.path.join(out_path, f'chr{chr}', f'VarInfo_chr{chr}_{i+1}.csv')
    code += f' variant_vcf '
    code += os.path.join(db_path, f'chr{chr}_{i+1}.csv')
    code += ' > '
    code += os.path.join(out_path, f'chr{chr}', f'Anno_chr{chr}_{i+1}.csv')
    subprocess.run(code, shell=True)


def do_annotation(vcf_mt, annot, reference_genome='GRCh37'):
    """
    Incorporating annotation in a Table to MatrixTable

    Parameters:
    ------------
    vcf_mt: a MatrixTable of VCF
    annot: a Table of functional annotation

    Returns:
    ---------
    vcf_mt: a MatrixTable of annotated VCF

    """
    split_varinfo = annot.VarInfo.split('-')
    chromosome = split_varinfo[0]
    position = hl.int(split_varinfo[1])
    ref_allele = split_varinfo[2]
    alt_allele = split_varinfo[3]

    annot = annot.annotate(locus=hl.locus(chromosome, position, reference_genome=reference_genome), 
                         alleles=[ref_allele, alt_allele])
    annot = annot.key_by('locus', 'alleles')
    annot = annot.drop('VarInfo')
    vcf_mt = vcf_mt.annotate_rows(fa=annot[vcf_mt.locus, vcf_mt.alleles])

    return vcf_mt


def convert_datatype(annot):
    """
    Converting numerical columns to float, imputing missing values as NaN

    Parameters:
    ------------
    anno: a Table of functional annotation

    Returns:
    ---------
    anno: a Table of functional annotation

    """
    annot = annot.annotate(
        apc_conservation = _impute_missing(annot.apc_conservation),
        apc_epigenetics = _impute_missing(annot.apc_epigenetics),
        apc_epigenetics_active = _impute_missing(annot.apc_epigenetics_active),
        apc_epigenetics_repressed = _impute_missing(annot.apc_epigenetics_repressed),
        apc_epigenetics_transcription = _impute_missing(annot.apc_epigenetics_transcription),
        apc_local_nucleotide_diversity = _impute_missing(annot.apc_local_nucleotide_diversity),
        apc_mappability = _impute_missing(annot.apc_mappability),
        apc_protein_function = _impute_missing(annot.apc_protein_function), 
        apc_transcription_factor = _impute_missing(annot.apc_transcription_factor),
        fathmm_xf = _impute_missing(annot.fathmm_xf),
        linsight = _impute_missing(annot.linsight),        
        cadd_phred = _impute_missing(annot.cadd_phred)                                                    
    )
    return annot


def _impute_missing(field):
    """
    Setting missing values as NA and converting string to float64

    """
    return hl.if_else(field == '', hl.missing(hl.tfloat64), hl.float(field))


def check_input(args, log):
    # required arguments
    if args.vcf is None:
        raise ValueError('--vcf is required')
    if args.favor_db is None:
        raise ValueError('--favor-db is required')
    if args.xsv is None:
        raise ValueError('--xsv is required')
    
    # required files must exist
    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"{args.vcf} does not exist")
    if not os.path.exists(args.favor_db):
        raise FileNotFoundError(f"{args.favor_db} does not exist")
    
    # optional arguments
    if args.threads is None:
        args.threads = 1
    
    # process arguments
    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    
    return geno_ref


def run(args, log):
    # check input and init
    geno_ref = check_input(args, log)
    log.info(f'Set {geno_ref} as the reference.')
    hl.init(quiet=True, default_reference=geno_ref)

    # convert VCF to MatrixTable
    log.info(f'Read VCF from {args.vcf}')
    vcf_mt = hl.import_vcf(args.vcf)

    # keep idvs
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
        vcf_mt = extract_idvs(vcf_mt, keep_idvs)
        
    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} variants in --extract.")
        vcf_mt = extract_snps(vcf_mt, keep_snps)

    # read and process snp info
    snp_info = vcf_mt.rows().select().to_pandas()
    snp_info = process_snp_info(snp_info)

    # partition chr and annotate each piece
    log.info(f'Read chromosome partition info from misc/FAVORdatabase_chrsplit.csv')
    db_info = pd.read_csv('misc/wgs/FAVORdatabase_chrsplit.csv')
    chr = snp_info['CHR'][0]
    db_info = db_info.loc[db_info['Chr'] == chr]

    subprocess.run(f"mkdir {os.path.join(args.out, f'chr{chr}')}", shell=True)
    log.info(f"Partition the chromosome and save to {os.path.join(args.out, f'chr{chr}')}")
    start = find_loc(snp_info['POS'], db_info.loc[db_info.index[0], 'Start_Pos'])
    if start == -1:
        start = 0
    for i in range(db_info.shape[0]):
        end = find_loc(snp_info['POS'], db_info.loc[db_info.index[i], 'End_Pos'])
        snp_info_i = snp_info.iloc[start: end, :].copy()
        snp_info_i['VarInfo'] = snp_info_i.apply(lambda x: f"{x['CHR']}-{x['POS']}-{x['REF']}-{x['ALT']}", axis=1)
        snp_info_i['VarInfo'].to_csv(os.path.join(args.out, f'chr{chr}', f'VarInfo_chr{chr}_{i+1}.csv'),
                          index=None)
        start = end
    
    log.info(f'Extract functional annotations for each block using {args.threads} threads ...')
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for i in range(db_info.shape[0]):
        #     extract_annotation(args.xsv, chr, i, args.out, args.favor_db)
            executor.submit(extract_annotation, args.xsv, chr, i, args.out, args.favor_db)

    # merge pieces
    log.info('Merge separate blocks.')
    merge_command = f'{args.xsv} cat rows '
    all_annot_dir = [os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}_{i+1}.csv') for i in range(db_info.shape[0])]
    merge_command += ' '.join(all_annot_dir)
    merge_command += ' > '
    merge_command += os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}.csv')
    subprocess.run(merge_command, shell=True)

    # subset
    log.info('Extract a subset of annotations.')
    annot_column = [1, 8, 9, 10, 11, 12, 15, 16, 19, 23] + list(range(25, 37))
    annot_column_xsv = ','.join([str(x) for x in annot_column])
    subset_command = f'{args.xsv} select {annot_column_xsv} '
    subset_command += os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}.csv')
    subset_command += ' > '
    subset_command += os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}_STAARpipeline.csv')
    subprocess.run(subset_command, shell=True)
    
    # read and add annotation to MatrixTable
    log.info('Annotate the genotypes.')
    annot = hl.import_table(os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}_STAARpipeline.csv'),
                           impute=True, delimiter=',', quote='"')
    annot_colnames = list(annot.row_value.keys())
    annot = annot.rename({annot_colnames[1]: 'apc_conservation',
                        annot_colnames[6]: 'apc_local_nucleotide_diversity',
                        annot_colnames[8]: 'apc_protein_function'})
    annot = convert_datatype(annot)
    vcf_mt = do_annotation(vcf_mt, annot, geno_ref)

    # save the MatrixTable
    vcf_mt.write(f'{args.out}_annotated_vcf.mt', overwrite=True)
    log.info(f'Write annotated VCF to MatrixTable {args.out}_annotated_vcf.mt')
