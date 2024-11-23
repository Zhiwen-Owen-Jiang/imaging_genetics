from heig.wgs.utils import (
    GProcessor,
    init_hail,
    process_range,
)


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
    log.info(f"Set QC mode as {args.qc_mode}")

    start_chr, start_pos, end_pos = process_range(args.range)

    return start_chr, start_pos, end_pos

def run(args, log):
    chr, start, end = check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # read genotype data
    if args.geno_mt is not None:
        log.info(f"Read MatrixTable from {args.geno_mt}")
        read_func = GProcessor.read_matrix_table
        data_path = args.geno_mt
    elif args.bfile is not None:
        log.info(f"Read bfile from {args.bfile}")
        read_func = GProcessor.import_plink
        data_path = args.bfile
    elif args.vcf is not None:
        log.info(f"Read VCF from {args.vcf}")
        read_func = GProcessor.import_vcf
        data_path = args.vcf

    gprocessor = read_func(
                data_path,
                grch37=args.grch37,
                hwe=args.hwe,
                variant_type=args.variant_type,
                maf_min=args.maf_min,
                maf_max=args.maf_max,
                call_rate=args.call_rate,
    )

    # do preprocessing
    log.info(f"Processing genotype data ...")
    gprocessor.extract_exclude_snps(args.extract, args.remove)
    gprocessor.keep_remove_idvs(args.keep, args.remove)
    gprocessor.do_processing(mode=args.qc_mode)
    gprocessor.extract_gene(chr=chr, start=start, end=end)
    gprocessor.check_valid()

    # save
    gprocessor.snps_mt.write(args.out, overwrite=True)
    log.info(f"\nSave genotype data at {args.out}")