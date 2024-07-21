import os
import shutil
import hail as hl
import heig.input.dataset as ds
from heig.wgs.utils import GProcessor, pandas_to_table


"""
TODO: consider providing more preprocessing options? such as --chr

"""


def config(args):
    spark_conf = {
        'spark.driver.memory': f'{args.mem}g',
        'spark.executor.memory': f'{args.mem}g'
    }
    hl.init(quiet=True, spark_conf=spark_conf, local=f'local[{args.threads}]',
    skip_logging_configuration=True)


def check_input(args, log):
    # required arguments
    if args.ldrs is None:
        raise ValueError('--ldrs is required')
    if args.covar is None:
        raise ValueError('--covar is required')
    if args.bfile is None and args.geno_mt is None:
        raise ValueError('either --bfile or --geno-mt is required')
    elif args.bfile is not None and args.geno_mt is not None:
        log.info('WARNING: --bfile is ignored if --geno-mt is provided')
        args.bfile = None
    
    # required files must exist
    if not os.path.exists(args.ldrs):
        raise FileNotFoundError(f"{args.ldrs} does not exist")
    if not os.path.exists(args.covar):
        raise FileNotFoundError(f"{args.covar} does not exist")
    if args.bfile is not None:
        for suffix in ['.bed', '.fam', '.bim']:
            if not os.path.exists(args.bfile + suffix):
                raise FileNotFoundError(f'{args.bfile + suffix} does not exist')
    if args.geno_mt is not None and not os.path.exists(args.geno_mt):
        raise FileNotFoundError(f"{args.geno_mt} does not exist")

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    
    if args.maf_min is not None:
        if args.maf_min > 0.5 or args.maf_min < 0:
            raise ValueError('--maf-min must be greater than 0 and less than 0.5')
    else:
        args.maf_min = 0
    
    if args.variant_type is None:
        args.variant_type = 'variant'
    else:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {'snv', 'variant', 'indel'}:
            raise ValueError("--variant-type must be one of ('variant', 'snv', 'indel')")
        
    if args.maf_max is None:
        args.maf_max = 0.5
    elif args.maf_max > 0.5 or args.maf_max <= 0 or args.maf_max <= args.maf_min:
        raise ValueError(('--maf-max must be greater than 0, less than 0.5, '
                          'and greater than --maf-min'))
    
    # process arguments
    if args.range is not None:
        try:
            start, end = args.range.split(',')
            start_chr, start_pos = [int(x) for x in start.split(':')]
            end_chr, end_pos = [int(x) for x in end.split(':')]
        except:
            raise ValueError(
                '--range should be in this format: <CHR>:<POS1>,<CHR>:<POS2>')
        if start_chr != end_chr:
            raise ValueError((f'starting with chromosome {start_chr} '
                                f'while ending with chromosome {end_chr} '
                                'is not allowed'))
        if start_pos > end_pos:
            raise ValueError((f'starting with {start_pos} '
                                f'while ending with position is {end_pos} '
                                'is not allowed'))
    else:
        start_chr, start_pos, end_pos = None, None, None

    temp_path = 'temp'
    i = 0
    while os.path.exists(temp_path):
        temp_path += str(i)
        i += 1

    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference genome.')
    
    return start_chr, start_pos, end_pos, temp_path, geno_ref


def do_gwas(snps_mt, n_ldrs, n_covar):
    pheno_list = [snps_mt.ldrs[i] for i in range(n_ldrs)]
    covar_list = [snps_mt.covar[i] for i in range(n_covar)]

    gwas = hl.linear_regression_rows(
        y=pheno_list, x=snps_mt.GT.n_alt_alleles(),
        covariates=covar_list, pass_through=[snps_mt.rsid]
    )

    gwas = gwas.annotate(chr=gwas.locus.contig,
                         pos=gwas.locus.position,
                         ref_allele=gwas.alleles[0],
                         alt_allele=gwas.alleles[1])
    gwas = gwas.key_by() 
    gwas = gwas.drop(*['locus', 'alleles', 'y_transpose_x', 'sum_x'])
    # gwas = gwas.drop(*['locus', 'alleles', 'n'])
    gwas = gwas.select('chr', 'pos', 'rsid', 'ref_allele', 'alt_allele',
                       'n_called', 'beta', 'standard_error', 't_stat',
                       'p_value')

    return gwas


def run(args, log):
    # check input and configure hail
    chr, start, end, temp_path, geno_ref = check_input(args, log)

    # read LDRs and covariates
    log.info(f'Read LDRs from {args.ldrs}')
    ldrs = ds.Dataset(args.ldrs)
    log.info(f'{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.')
    if args.n_ldrs is not None:
        ldrs.data = ldrs.data.iloc[:, :args.n_ldrs]
        if ldrs.data.shape[1] > args.n_ldrs:
            log.info(f'WARNING: --n-ldrs greater than #LDRs, use all LDRs.')
        else:
            log.info(f'Keep the top {args.n_ldrs} LDRs.')        

    log.info(f'Read covariates from {args.covar}')
    covar = ds.Covar(args.covar, args.cat_covar_list)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None
    common_ids = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs, single_id=True)

    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs in --extract.")
    else:
        keep_snps = None

    # read genotype data
    hl.init(quiet=True)
    hl.default_reference = geno_ref

    if args.bfile is not None:
        log.info(f'Read bfile from {args.bfile}')
        gprocessor = GProcessor.import_plink(args.bfile, geno_ref,
                                             variant_type=args.variant_type, 
                                             maf_min=args.maf_min, maf_max=args.maf_max)
    elif args.geno_mt is not None:
        log.info(f'Read genotype data from {args.geno_mt}')
        gprocessor = GProcessor.read_matrix_table(args.geno_mt, geno_ref,
                                                  variant_type=args.variant_type, 
                                                  maf_min=args.maf_min, maf_max=args.maf_max)
   
    log.info(f"Processing genetic data ...")
    gprocessor.extract_snps(keep_snps)
    gprocessor.extract_idvs(common_ids)
    gprocessor.do_processing(mode='gwas')
    if chr is not None and start is not None:
        # TODO: add more options
        gprocessor.extract_gene(chr=chr, start=start, end=end)
    
    if not args.not_save_genotype_data:
        log.info(f'Save preprocessed genotype data to {temp_path}')
        gprocessor.save_interim_data(temp_path)
    gprocessor.check_valid()
    
    try:
        # extract common subjects and align data
        snps_mt_ids = gprocessor.subject_id()
        ldrs.to_single_index()
        covar.to_single_index()
        ldrs.keep(snps_mt_ids)
        covar.keep(snps_mt_ids)
        covar.cat_covar_intercept()
        log.info(f'{len(common_ids)} common subjects in the data.')
        log.info(f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept).")

        covar_table = pandas_to_table(covar.data, temp_path)
        ldrs_table = pandas_to_table(ldrs.data, temp_path)

        # annotate ldrs and covar to snps_mt
        gprocessor.annotate_cols(ldrs_table, 'ldrs')
        gprocessor.annotate_cols(covar_table, 'covar')

        # gwas
        n_ldrs = ldrs.data.shape[1]
        n_covar = covar.data.shape[1]
        log.info(f'Doing GWAS for {n_ldrs} LDRs ...')
        gwas = do_gwas(gprocessor.snps_mt, n_ldrs, n_covar)

        # save gwas results
        gwas.export(f"{args.out}.txt.bgz")
        log.info(f"Save GWAS results to {args.out}.txt.bgz")
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')