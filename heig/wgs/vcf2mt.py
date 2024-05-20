import os
import subprocess
import hail as hl
import pandas as pd


"""
1. Save a raw VCF file to a MatrixTable. The VCF file should include all
subjects for a single chromosome
2. Generate a variant list to be annotated
3. Extract annotation for each piece
4. Merge and subset the annotation

"""


def process_config(spark_config):
    pass


def process_snp_info(snp_info):
    chr_list = snp_info['locus'].apply(lambda x: x.contig)
    if len(set(chr_list)) != 1:
        raise ValueError('the VCF file must contain only one chromosome')
    pos_list = snp_info['locus'].apply(lambda x: x.position)
    ref_list = snp_info['alleles'].apply(lambda x: x[0])
    alt_list = snp_info['alleles'].apply(lambda x: x[1])

    snp_df = pd.DataFrame({'CHR': chr_list, 'POS': pos_list,
                           'REF': ref_list, 'ALT': alt_list})
    return snp_df


def find_loc(num_list, target):
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] == target:
            return mid
        elif num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return r


def do_annotation(xsv, out, chr, i, db_path):
    code = f'{xsv} join --left VarInfo '
    code += os.path.join(out, f'chr{chr}', f'VarInfo_chr{chr}_{i+1}.csv')
    code += f' variant_vcf '
    code += os.path.join(db_path, f'chr{chr}_{i+1}.csv')
    code += ' > '
    code += os.path.join(out, f'chr{chr}', f'Anno_chr{chr}_{i+1}.csv')
    subprocess.run(code, shell=True)


def run(args, log):
    spark_config = process_config(args.spark_config)
    hl.init(spark_conf=spark_config)  # TODO: investigate

    # convert VCF to MatrixTable
    log.info(f'Read VCF from {args.vcf}')
    vcf_mt = hl.import_vcf(args.vcf)

    # read and process snp info
    snp_info = vcf_mt.rows().select().to_pandas()
    snp_info = process_snp_info(snp_info)

    # partition chr and annotation each piece
    log.info(f'Read chromosome partition info from misc/FAVORdatabase_chrsplit.csv')
    db_info = pd.read_csv('misc/FAVORdatabase_chrsplit.csv')
    chr = snp_info['CHR'][0]
    db_info = db_info.loc[db_info['Chr'] == chr]

    log.info(
        f'Partition the chromosome and save to {os.path.join(args.out, chr)}')
    start = find_loc[snp_info['POS'],
                     db_info.loc[db_info.index[0], 'Start_Pos']]
    for i in range(db_info.shape[0]):
        end = find_loc[snp_info['POS'],
                       db_info.loc[db_info.index[i], 'End_Pos']]
        snp_info_i = snp_info_i.iloc[start: end, :]
        snp_info_i.to_csv(os.path.join(args.out, chr, f'VarInfo_chr{chr}_{i+1}.csv'),
                          index=None)
        do_annotation(args.xsv, args.out, chr, i, args.db_path)
        start = end

    # merge pieces
    merge_command = f'{args.xsv} cat rows '
    all_anno_dir = [os.path.join(
        args.out, f'chr{chr}', f'Anno_chr{chr}_{i+1}.csv') for i in range(db_info.shape[0])]
    merge_command += ' '.join(all_anno_dir)
    merge_command += ' > '
    merge_command += os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}.csv')
    subprocess.run(merge_command, shell=True)

    # subset
    anno_column = [1, 8, 9, 10, 11, 12, 15, 16, 19, 23] + list(range(25, 37))
    anno_column_xsv = ','.join(anno_column)
    subset_command = f'{args.xsv} select {anno_column_xsv} '
    subset_command += os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}.csv')
    subset_command += ' > '
    subset_command += os.path.join(args.out,
                                   f'chr{chr}', f'Anno_chr{chr}_STAARpipeline.csv')
    subprocess.run(subset_command, shell=True)
    
    # read and add annotation to MatrixTable
    # TODO: check column types
    # anno = pd.read_csv(os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}_STAARpipeline.csv'))
    anno = hl.import_table(os.path.join(args.out, f'chr{chr}', f'Anno_chr{chr}_STAARpipeline.csv'),
                           impute=True, delimiter=',')
    anno_colnames = list(anno.row_value.keys())
    anno = anno.rename({anno_colnames[1]: 'apc_conservation',
                        anno_colnames[6]: 'apc_local_nucleotide_diversity',
                        anno_colnames[8]: 'apc_protein_function'},
                        axis=1)
    # TODO: check the key of annotation

    vcf_mt.annotate_rows(fa=anno)

    # save the MatrixTable
    vcf_mt.write(f'{args.out}_annotated_vcf.mt', overwrite=True)
    log.info(f'Write annotated VCF to MatrixTable {args.out}_annotated_vcf.mt')
