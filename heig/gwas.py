import hail as hl
import pandas as pd
import heig.input.dataset as ds


def config(args):
    spark_conf = {
        'spark.driver.memory': f'{args.mem}g',
        'spark.executor.memory': f'{args.mem}g'
    }
    hl.init(quiet=True, spark_conf=spark_conf, local=f'local[{args.threads}]',
    skip_logging_configuration=True)


def read_process_idvs(fam_dir):
    fam = pd.read_csv(fam_dir, sep='\s+', header=None,
                      names=['FID', 'IID', 'FATHER', 'MOTHER', 'GENDER', 'TRAIT'],
                      dtype={'FID': str, 'IID': str})
    fam = fam.set_index(['FID', 'IID'])

    return fam


def check_input(args, log):
    if args.ldrs is None:
        raise ValueError('--ldrs is required')
    if args.covar is None:
        raise ValueError('--covar is required')
    if args.mem is None:
        args.mem = 8
    if args.threads is None:
        args.threads = 4
    if args.grch37:
        args.ref_genome = 'GRCh37'
    else:
        args.ref_genome = 'GRCh38'

    return args


def run(args, log):
    # check input and configure hail
    args = check_input(args, log)
    config(args)

    # read ldrs and covar and --keep
    log.info(f'Read LDRs from {args.ldrs}')
    ldrs = ds.Dataset(args.ldrs)
    log.info(f'Read covariates from {args.covar}')
    covar = ds.Covar(args.covar, args.cat_covar_list)

    if args.n_ldrs is not None:
        ldrs.data = ldrs.data.iloc[:, :args.n_ldrs]

    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
    else:
        keep_idvs = None

    # fam = read_process_idvs(args.bfile + '.fam')
    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, keep_idvs)
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    covar.to_single_index()
    ldrs.keep(common_idxs)
    ldrs.to_single_index()
    
    covar.data.to_csv(f'{args.out}_covar.tb', sep='\t', index=None)
    covar_table = hl.import_table(f'{args.out}_covar.tb', key='IID', impute=True, types={'IID': hl.tstr})
    ldrs.data.to_csv(f'{args.out}_ldrs.tb', sep='\t', index=None)
    ldrs_table = hl.import_table(f'{args.out}_ldrs.tb', key='IID', impute=True, types={'IID': hl.tstr})

    # bed
    if args.bfile is not None:
        log.info(
            f'Read bfile from {args.bfile} and save the MatrixTable to {args.bfile}.mt')
        hl.import_plink(bed=args.bfile + '.bed',
                        bim=args.bfile + '.bim',
                        fam=args.bfile + '.fam',
                        reference_genome=args.ref_genome,
                        ).write(f"{args.out}_geno.mt")
        snps = hl.read_matrix_table(f"{args.out}_geno.mt")
    elif args.geno_mt is not None:
        log.info(f'Read genotype data from {args.geno_mt}')
        snps = hl.read_matrix_table(f"{args.geno_mt}")

    # filtering SNPs
    if args.maf_min is not None:
        log.info(f'Remove SNPs with an MAF less than {args.maf_min}')
        snps = hl.variant_qc(snps)
        snps = snps.filter_rows(snps.variant_qc.AF[1] > args.maf_min)
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        snps = snps.filter_rows(hl.literal(
            keep_snps['SNP']).contains(snps.rsid))
    log.info(f'{snps.count_rows()} SNPs for doing GWAS.')

    # keep subjects
    # snps = snps.filter_cols(hl.literal(ldrs_table.s).contains(snps.s)))

    # append ldrs and covar to snps
    snps = snps.annotate_cols(ldrs=ldrs_table[snps.s])
    snps = snps.annotate_cols(covar=covar_table[snps.s])

    # gwas
    pheno_list = [snps.ldrs[i] for i in range(len(ldrs_table.row.dtype) - 1)]
    covar_list = [snps.covar[i] for i in range(1, covar.data.shape[1] - 1)]
    covar_list.append(1)

    log.info((f'Doing GWAS for {ldrs.data.shape[1]} LDRs in parallel '
              f'using {args.threads} threads ...'))
    gwas = hl.linear_regression_rows(y=pheno_list,
                                     x=snps.GT.n_alt_alleles(),
                                     covariates=covar_list,
                                     pass_through=['rsid'])

    # save gwas results
    log.info(f"Save GWAS results to {args.out}.txt.bgz")
    gwas.export(f"{args.out}.txt.bgz")
