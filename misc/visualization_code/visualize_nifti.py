import argparse
import pickle
import h5py
import nibabel as nb
import numpy as np
import pandas as pd


def get_nearest_point(idxs_res, idxs_to_impute):
    nearest_point = {
        tuple(idx): _get_nearest_point(idx, idxs_res) for idx in idxs_to_impute
    }
    return nearest_point


def _get_nearest_point(target, coord):
    dis = [np.sum((target - idx) ** 2) for idx in coord]
    return np.argmin(dis)


def main(args):
    # reading template
    print(f"Read mask from {args.mask}")
    temp = nb.load(args.mask)
    temp_data = temp.get_fdata()

    # reading coordinate
    if args.coord is not None:
        coord = np.loadtxt(args.coord, dtype=int)
        print(f"Read coordinates from {args.coord}")
    elif args.coord_h5 is not None:
        with h5py.File(args.coord_h5, "r") as file:
            coord = file["coord"][:]
        print(f"Read coordinates from {args.coord_h5}")
    else:
        raise ValueError("Either --coord or --coord-h5 is required.")

    # extracting indices in the mask that need to impute
    idxs_temp = np.where(temp_data == 1)
    idxs_temp = set(zip(*idxs_temp))
    idxs_res = tuple(zip(*coord.T))
    idxs_to_impute = idxs_temp.difference(idxs_res)
    idxs_res = np.array(idxs_res)
    idxs_to_impute = np.array(list(idxs_to_impute))
    print("Extracted indices that need to impute.")

    # reading result file
    print(f"Read results from {args.res}")
    res_file = pd.read_csv(args.res, sep="\s+")

    # getting results and filling into the mask
    if hasattr(res_file, args.col_name):
        res = res_file[args.col_name] + args.offset
    else:
        raise ValueError(f"{args.col_name} cannot be found in {args.res}")
    temp_data[tuple(zip(*coord))] = res
    print(f"Filled results to the mask.")

    # computing nearest points and imputing
    if args.nn is None:
        print(f"Doing imputation by nearest neighbor algorithm ...")
        nearest_point = get_nearest_point(idxs_res, idxs_to_impute)
        pickle.dump(nearest_point, open(f"{args.out}_nn.dat", "wb"))
        print(f"Save nearest neighbor information to {args.out}_nn.dat")
    else:
        print(f"Read nearest neighbor information from {args.nn} and impute the mask.")
        nearest_point = pickle.load(open(args.nn, "rb"))
    for target, point in nearest_point.items():
        temp_data[target] = res[point]

    # output imputed image
    print(f"Save imputed image to {args.out}.nii.gz")
    new_img = nb.Nifti1Image(
        temp_data,
        temp.affine,
    )
    nb.save(new_img, f"{args.out}.nii.gz")


parser = argparse.ArgumentParser()
parser.add_argument("--coord", help="a white space-delimited coordinate file")
parser.add_argument("--mask", help="a mask file (e.g., .nii.gz) as template")
parser.add_argument(
    "--offset", type=int, help="a fixed number to ensure every result greater than 0."
)
parser.add_argument("--res", help="a white space-delimited result file")
parser.add_argument(
    "--col-name", help="which column in the result file you want to visualize"
)
parser.add_argument("--out", help="output (prefix)")
parser.add_argument("--nn", help="a dat file for nearest neighbor information")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
