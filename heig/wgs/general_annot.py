import os
import shutil
import numpy as np
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.staar import VariantSetTest,  prepare_vset_test
from heig.wgs.utils import *
from heig.wgs.coding import format_output



def single_gene_analysis(snps_mt, vset_test, annot_names, log):
    """
    Single gene analysis

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated geno
    vset_test: an instance of VariantSetTest
    log: a logger

    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)

    """
    # individual analysis
    cate_pvalues = dict()
    maf, is_rare, vset = prepare_vset_test(snps_mt)

    annot = snps_mt.fa.select(*annot_names).collect()
    annot = np.array(
        [
            [getattr(row, col) for col in annot_names]
            for row in annot
        ]
    )
    vset_test.input_vset(vset, maf, is_rare, annot, annot_transform=False)
    log.info(
        f"Doing analysis for {vset_test.n_variants} variants using {len(annot_names)} annotation(s) ..."
    )
    pvalues = vset_test.do_inference(annot_names)
    cate_pvalues['annotation'] = {
        "n_variants": vset_test.n_variants,
        "pvalues": pvalues,
    } # solve 'annotation' problem

    return cate_pvalues


def check_input(args, log):
    # required arguments
    if args.geno_mt is None:
        raise ValueError("--geno-mt is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")
    if args.annot_name is None:
        raise ValueError("--annot-name is required")
    
    args.annot_name = args.annot_name.split(',')

    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")
    else:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {"snv", "variant", "indel"}:
            raise ValueError(
                "--variant-type must be one of ('variant', 'snv', 'indel')"
            )

    if args.maf_max is None:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")
    elif args.maf_max > 0.5 or args.maf_max <= 0 or args.maf_max <= args.maf_min:
        raise ValueError(
            (
                "--maf-max must be greater than 0, less than 0.5, "
                "and greater than --maf-min"
            )
        )

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    
    # process arguments
    if args.range is not None:
        start_chr, start_pos, end_pos = process_range(args.range)
        return start_chr, start_pos, end_pos
    else:
        return None, None, None


def run(args, log):
    chr, start, end = check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # reading data and selecting voxels and LDRs
    log.info(f"Read null model from {args.null_model}")
    null_model = NullModel(args.null_model)
    null_model.select_voxels(args.voxel)
    null_model.select_ldrs(args.n_ldrs)

    # read loco preds
    try:
        if args.loco_preds is not None:
            log.info(f"Read LOCO predictions from {args.loco_preds}")
            loco_preds = LOCOpreds(args.loco_preds)
            loco_preds.select_ldrs(args.n_ldrs)
            if loco_preds.n_ldrs != null_model.n_ldrs:
                raise ValueError(
                    (
                        "inconsistent dimension in LDRs and LDR LOCO predictions. "
                        "Try to use --n-ldrs"
                    )
                )
            common_ids = ds.get_common_idxs(
                null_model.ids,
                loco_preds.ids,
                args.keep,
                single_id=True,
            )
        else:
            common_ids = ds.get_common_idxs(null_model.ids, args.keep, single_id=True)

        # read genotype data
        log.info(f"Reading genotype data from {args.geno_mt}")
        gprocessor = GProcessor.read_matrix_table(
            args.geno_mt,
            grch37=args.grch37,
            variant_type=args.variant_type,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            mac_thresh=args.mac_thresh
        )

        # do preprocessing
        log.info(f"Processing genotype data ...")
        gprocessor.extract_snps(args.extract)
        gprocessor.extract_idvs(common_ids)
        gprocessor.do_processing(mode="wgs")
        gprocessor.extract_gene(chr=chr, start=start, end=end)

        # save processsed data for faster analysis
        if not args.not_save_genotype_data:
            temp_path = get_temp_path()
            log.info(f"Save preprocessed genotype data to {temp_path}")
            gprocessor.save_interim_data(temp_path)

        # gprocessor.check_valid()
        if chr is None:
            chr, start, end = gprocessor.extract_range()
        # extract and align subjects with the genotype data
        snps_mt_ids = gprocessor.subject_id()
        null_model.keep(snps_mt_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(snps_mt_ids)} common subjects in the data.")
        log.info(
            (f"{null_model.covar.shape[1]} fixed effects in the covariates (including the intercept) "
             "after removing redundant effects.\n")
        )

        if args.loco_preds is not None:
            loco_preds.keep(snps_mt_ids)
        else:
            loco_preds = None

        # single gene analysis
        vset_test = VariantSetTest(
            null_model.bases, null_model.resid_ldr, null_model.covar, chr, loco_preds
        )
        cate_pvalues = single_gene_analysis(
            gprocessor.snps_mt,
            vset_test,
            args.annot_name,
            log,
        )

        # format output
        for cate, cate_results in cate_pvalues.items():
            cate_output = format_output(
                cate_results["pvalues"],
                chr,
                start,
                end,
                cate_results["n_variants"],
                null_model.voxel_idxs,
                cate,
            )
            out_path = f"{args.out}_chr{chr}_start{start}_end{end}_{cate}.txt"
            cate_output.to_csv(
                out_path,
                sep="\t",
                header=True,
                na_rep="NA",
                index=None,
                float_format="%.5e",
            )
            log.info(f"\nSave results for {cate} to {out_path}")
    finally:
        if "temp_path" in locals():
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                log.info(f"Removed preprocessed genotype data at {temp_path}")
        if args.loco_preds is not None:
            loco_preds.close()