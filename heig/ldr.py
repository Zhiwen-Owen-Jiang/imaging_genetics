import h5py
import concurrent.futures
import numpy as np
import pandas as pd
import heig.input.dataset as ds
from collections import defaultdict
from heig.fpca import image_reader
from heig.utils import inv


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S/n = S'S - S'X(X'X)^{-1}X'S/n,
    where I is the identity matrix,
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    ldr (n, r): low-dimension representaion of imaging data
    covar (n, p): covariates, including the intercept

    Returns:
    ---------
    ldr_cov: variance-covariance matrix of LDRs

    """
    n = ldr.shape[0]
    inner_ldr = np.dot(ldr.T, ldr)
    inner_covar = np.dot(covar.T, covar)
    inner_covar_inv = inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    ldr_cov = (inner_ldr - part2) / n
    ldr_cov = ldr_cov.astype(np.float32)

    return ldr_cov


def image_recovery_quality(images, ldrs, bases):
    """
    Computing correlation between raw images and reconstructed images

    Parameters:
    ------------
    images: a np.array of normalized raw images (n, N)
    ldrs: a np.array of constructed LDRs (n, r)
    bases: a np.array of corresponding bases (r, N)

    Returns:
    ---------
    corr: a np.array of correlation coefficients between raw and reconstructed images

    """
    rec_images = np.dot(bases, ldrs.T)
    rec_images = (rec_images - np.mean(rec_images, axis=0)) / np.std(rec_images, axis=0)
    corr = np.mean(images * rec_images, axis=0)

    return corr


def construct_ldr_batch(
    images_, start_idx, end_idx, bases, alt_n_ldrs_list, rec_corr, ldrs
):
    """
    Construting LDRs in batch

    Parameters:
    ------------
    images_: a np.array of raw images (n1, N)
    start_idx: start index
    end_idx: end index
    bases: a np.array of bases (N, r)
    alt_n_ldrs_list: a list of alternative number of LDRs
    rec_corr: a dict of reconstruction correlation
    ldrs: a np.array of LDRs (n1, r)

    """
    ldrs_ = np.dot(images_, bases)
    ldrs[start_idx:end_idx] = ldrs_
    images_ = images_.T
    images_ = (images_ - np.mean(images_, axis=0)) / np.std(images_, axis=0)

    for alt_n_ldrs in alt_n_ldrs_list:
        image_rec_corr = image_recovery_quality(
            images_, ldrs_[:, :alt_n_ldrs], bases[:, :alt_n_ldrs]
        )
        rec_corr[alt_n_ldrs][start_idx:end_idx] = image_rec_corr


def print_alt_corr(rec_corr, log):
    """
    Printing a table of reconstruction correlation
    using varying numbers of LDRs

    """
    max_key_len = max(len(str(key)) for key in rec_corr.keys())
    max_val_len = max(len(str(value)) for value in rec_corr.values())
    max_len = max([max_key_len, max_val_len])
    keys_str = "  ".join(f"{str(key):<{max_len}}" for key in rec_corr.keys())
    values_str = "  ".join(f"{str(value):<{max_len}}" for value in rec_corr.values())

    log.info(
        "Mean correlation between reconstructed images and raw images using varying numbers of LDRs:"
    )
    log.info(keys_str)
    log.info(values_str)

    max_corr = max(rec_corr.values())
    max_n_ldrs = max(rec_corr.keys())
    if max_corr < 0.85:
        log.info(
            (
                f"Using {max_n_ldrs} LDRs can achieve a correlation coefficient of {max_corr}, "
                "which might be too low, consider increasing LDRs.\n"
            )
        )


def check_input(args):
    # required arguments
    if args.image is None:
        raise ValueError("--image is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.bases is None:
        raise ValueError("--bases is required")


def run(args, log):
    check_input(args)

    # read bases and extract top n_ldrs
    bases = np.load(args.bases)
    n_voxels, n_bases = bases.shape
    log.info(f"{n_bases} bases of {n_voxels} voxels (vertices) read from {args.bases}")

    if args.n_ldrs is not None:
        if args.n_ldrs <= n_bases:
            n_ldrs = args.n_ldrs
            bases = bases[:, :n_ldrs]
        else:
            raise ValueError("the number of bases is less than --n-ldrs")
    else:
        n_ldrs = n_bases

    # read images
    log.info(f"Read raw images from {args.image}")
    with h5py.File(args.image, "r") as file:
        images = file["images"]
        ids = file["id"][:]
        ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        if n_voxels != images.shape[1]:
            raise ValueError("the images and bases have different resolution")

        # read covariates
        log.info(f"Read covariates from {args.covar}")
        covar = ds.Covar(args.covar, args.cat_covar_list)

        # keep common subjects
        common_idxs = ds.get_common_idxs(ids, covar.data.index, args.keep)
        log.info(f"{len(common_idxs)} common subjects in these files.")

        # contruct ldrs
        ids_ = ids.isin(common_idxs)
        id_idxs = np.arange(len(ids))[ids_]
        ldrs = np.zeros((len(id_idxs), n_ldrs), dtype=np.float32)

        start_idx, end_idx = 0, 0
        rec_corr = defaultdict(lambda: np.zeros(len(id_idxs)))
        alt_n_ldrs_list = [int(n_ldrs * prop) for prop in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)]

        log.info(f"Constructing {n_ldrs} LDRs ...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads
        ) as executor:
            futures = []
            for images_ in image_reader(images, id_idxs):
                start_idx = end_idx
                end_idx += images_.shape[0]

                futures.append(
                    executor.submit(
                        construct_ldr_batch,
                        images_,
                        start_idx,
                        end_idx,
                        bases,
                        alt_n_ldrs_list,
                        rec_corr,
                        ldrs,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()

        for alt_n_ldrs, corr in rec_corr.items():
            rec_corr[alt_n_ldrs] = round(np.mean(corr), 2)

        print_alt_corr(rec_corr, log)

    # process covar
    covar.keep(common_idxs)
    covar.cat_covar_intercept()
    log.info(
        f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
    )

    # var-cov matrix of projected LDRs
    ldr_cov = projection_ldr(ldrs, np.array(covar.data))
    log.info(
        f"Removed covariate effects from LDRs and computed variance-covariance matrix.\n"
    )

    # save the output
    ldr_df = pd.DataFrame(ldrs, index=ids[ids_])
    ldr_df.to_csv(f"{args.out}_ldr_top{n_ldrs}.txt", sep="\t")
    np.save(f"{args.out}_ldr_cov_top{n_ldrs}.npy", ldr_cov)

    log.info(f"Save the raw LDRs to {args.out}_ldr_top{n_ldrs}.txt")
    log.info(
        (
            f"Save the variance-covariance matrix of covariate-effect-removed LDRs "
            f"to {args.out}_ldr_cov_top{n_ldrs}.npy"
        )
    )
