from heig.wgs.utils import init_hail, read_genotype_data


def check_input(args, log):
    if args.bfile is None and args.vcf is None and args.geno_mt is None:
        raise ValueError("--bfile or --vcf or --geno-mt is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    
    if args.geno_mt is not None:
        args.vcf, args.bfile = None, None
    if args.bfile is not None:
        args.vcf = None
    if args.qc_mode is None:
        args.qc_mode = 'gwas'
    log.info(f"Set QC mode as {args.qc_mode}.")


def run(args, log):
    check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # read genotype data
    gprocessor = read_genotype_data(args, log)

    # do preprocessing
    log.info(f"Processing genotype data ...")
    gprocessor.extract_exclude_locus(args.extract_locus, args.exclude_locus)
    gprocessor.extract_exclude_snps(args.extract, args.exclude)
    gprocessor.extract_chr_interval(args.chr_interval)
    gprocessor.keep_remove_idvs(args.keep, args.remove)
    gprocessor.do_processing(mode=args.qc_mode)
    gprocessor.check_valid()

    # save
    gprocessor.snps_mt.write(args.out, overwrite=True)
    log.info(f"\nSave genotype data at {args.out}")