import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
import heig.input.dataset as ds


"""
Threshold-free cluster enhancement (TFCE) analysis for 
significant/null pvalues

1. summarize significant results
2. summarize null results

TODO:
1. support more image format

"""

class TFCE:
    def __init__(self, coord, roi_mask, slices, h=2, E=0.5, dh=0.1):
        self.coord = coord
        self.roi_mask = roi_mask
        self.slices = slices
        self.h = h
        self.E = E
        self.dh = dh

    def _tfce(self, stat_map):
        """
        Compute Threshold-Free Cluster Enhancement (TFCE) on a statistical map.

        Parameters:
        - stat_map (np.ndarray): Input statistical map (2D or 3D).
        - h (float): Height exponent.
        - E (float): Extent exponent.
        - dh (float): Step size for integration.

        Returns:
        - tfce_map (np.ndarray): Enhanced TFCE statistical map.
        """
        tfce_map = np.zeros_like(stat_map)
        tfce_map[stat_map == 0.001] = 0.001
        non_zeros = stat_map[stat_map > 0.001]
        thresholds = np.arange(np.min(non_zeros), np.max(non_zeros), self.dh)

        for threshold in thresholds:
            # Binary mask of voxels exceeding the current threshold
            binarized_map = stat_map >= threshold

            # Label connected components
            labeled_clusters, num_clusters = label(binarized_map)

            for cluster_id in range(1, num_clusters + 1):
                cluster_size = np.sum(labeled_clusters == cluster_id)
                tfce_map[labeled_clusters == cluster_id] += (cluster_size ** self.E) * (threshold ** self.h) * self.dh

        return tfce_map
    
    def tfce(self, index, results):
        """
        Generating a cropped TFCE map

        Parameters:
        ------------
        index: a np.array of zero-based idxs
        results: a np.array of -log10 pvalues
        
        """
        all_res = np.ones(len(self.coord[0])) * 0.001
        all_res[index] = results
        stat_map = np.zeros(self.roi_mask.shape)
        stat_map[self.roi_mask] = all_res

        stat_map_crop = stat_map[self.slices]
        tfce_map = self._tfce(stat_map_crop)

        return tfce_map


def crop_image_with_margin(image, margin=1):
    """
    Crops a 2D or 3D image to the smallest bounding box containing nonzero values,
    while keeping a margin of at least 'margin' voxels.

    Parameters:
    - image (np.ndarray): Input 2D or 3D image.
    - margin (int): Number of voxels to leave as a margin.

    Returns:
    - slices (np.ndarray): voxel idxs of cropped image.
    """
    # Find nonzero voxel coordinates
    nonzero_coords = np.where(image != 0)
    
    # Get bounding box (min and max indices along each axis)
    min_indices = [max(np.min(axis) - margin, 0) for axis in nonzero_coords]
    max_indices = [min(np.max(axis) + margin + 1, image.shape[i]) for i, axis in enumerate(nonzero_coords)]
    
    # Crop image
    slices = tuple(slice(min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices))
    # cropped_image = image[slices]

    return slices


def nifti_coord_mask(coord_img_file):
    img = nib.load(coord_img_file)
    data = img.get_fdata()
    roi_mask = data > 0
    coord = np.stack(np.nonzero(data)).T
    coord = tuple(zip(*coord))
    slices = crop_image_with_margin(data)

    return coord, roi_mask, slices


def summarize_results(tfce, results_idx, variant_category, sig_thresh, tfce_thresh):
    gene = list()
    chr = list()
    start = list()
    end = list()
    n_variants = list()
    cmac = list()
    most_sig_pv = list()
    n_clusters = list()
    n_sig_voxels = list()
    cluster_info = list()
    max_tfce = list()

    for _, result_info in results_idx.iterrows():
        results = pd.read_csv(
            result_info['RESULT_FILE'], 
            sep='\t', 
            usecols=["INDEX", "MASK", "N_VARIANTS", "CMAC", "STAAR-O"]
        )
        results = results[
            (results["MASK"] == variant_category) & (results["STAAR-O"] < sig_thresh)
        ]

        if len(results) == 0:
            continue
        results["INDEX"] -= 1
        log_pvalues = -np.log10(results["STAAR-O"])
        tfce_res = tfce.tfce(results["INDEX"], log_pvalues)
        labeled_clusters, num_clusters = label(tfce_res > tfce_thresh)
        
        if num_clusters == 0:
            continue
        n_clusters.append(num_clusters - 1)
        max_tfce.append(np.max(tfce_res))
        n_sig_voxels.append(len(log_pvalues))

        voxels_in_cluster_list = list()
        for cluster in range(1, num_clusters + 1):
            voxels_in_cluster = np.where(labeled_clusters[labeled_clusters >= 0.001] == cluster)[0] + 1
            voxels_in_cluster_list.append(voxels_in_cluster)
        voxels_in_cluster = ';'.join([','.join(x.astype(str)) for x in voxels_in_cluster_list])
        cluster_info.append(voxels_in_cluster)
        
        gene.append(result_info['VARIANT_SET'])
        chr.append(result_info['CHR'])
        start.append(result_info['START'])
        end.append(result_info['END'])
        n_variants.append(results['N_VARIANTS'].to_list()[0])
        cmac.append(results['CMAC'].to_list()[0])
        most_sig_pv.append(results['STAAR-O'].min())

    results_summary = pd.DataFrame(
        {
            'GENE': gene,
            'CHR': chr,
            'START': start,
            'END': end,
            'CATEGORY': variant_category,
            'N_VARIANTS': n_variants,
            'CMAC': cmac,
            'MOST_SIG_PV': most_sig_pv,
            'MAX_TFCE': max_tfce,
            'N_SIG_VOXELS': n_sig_voxels,
            'N_CLUSTERS': n_clusters,
            'VOXELS_IN_EACH_CLUSTER': cluster_info,
        }
    )
    
    return results_summary


def summarize_null_results(tfce, null_assoc, sig_thresh):
    null_assoc = null_assoc[null_assoc["STAAR-O"] < sig_thresh]
    null_assoc["LOG10P"] = -np.log10(null_assoc["STAAR-O"])
    null_assoc["INDEX"] -= 1
    null_assoc_group = null_assoc.groupby(["SAMPLE_ID", "GENE_ID"])
    null_assoc_tfce = list()

    for _, null_assoc_ in null_assoc_group:
        tfce_res = tfce.tfce(null_assoc_["INDEX"], null_assoc_["LOG10P"])
        null_assoc_tfce.append(np.max(tfce_res))

    return np.array(null_assoc_tfce)
        

def check_input(args, log):
    if args.coord_dir is None:
        raise ValueError("--coord-dir is required")
    else:
        ds.check_existence(args.coord_dir)

    if args.results_idx is None and args.null_assoc is None:
        raise ValueError("--result-idx or --null-assoc is required")
    if args.results_idx is not None:
        args.results_idx = ds.parse_input(args.results_idx)
        for file in args.results_idx:
            ds.check_existence(file)
        if args.tfce_thresh is None:
            args.tfce_thresh = 0
            log.info("Set TFCE threshold as 0")
        if args.variant_category is None:
            raise ValueError("--variant-category is required")
        else:
            args.variant_category = args.variant_category.lower()
            if args.variant_category not in {
                    "plof",
                    "plof_ds",
                    "missense",
                    "disruptive_missense",
                    "synonymous",
                    "ptv",
                    "ptv_ds",
                }:
                raise ValueError(f"invalid variant category: {args.variant_category}")
        if args.tfce_thresh is None:
            raise ValueError("--tfce-thresh is required")
    if args.null_assoc is not None:
        ds.check_existence(args.null_assoc)
    if args.sig_thresh is None:
        args.sig_thresh = 2.5e-6
        log.info("Set significance threshold as 2.5e-6")


def run(args, log):
    check_input(args, log)

    log.info(f"Read MASK image from {args.coord_dir}")
    coord, roi_mask, slices = nifti_coord_mask(args.coord_dir)
    tfce = TFCE(coord, roi_mask, slices)

    if args.results_idx is not None:
        results_summary_list = list()
        for results_idx_file in args.results_idx:
            log.info(f"Read result index file from {results_idx_file}")
            results_idx = pd.read_csv(results_idx_file, sep='\t')

            results_summary = summarize_results(
                tfce,
                results_idx, 
                args.variant_category, 
                args.sig_thresh, 
                args.tfce_thresh
            )
            results_summary_list.append(results_summary)
        results_summary = pd.concat(results_summary_list, axis=0)
        results_summary.to_csv(f"{args.out}.txt", sep="\t", index=None)
        log.info(f"\nSaved result summary to {args.out}.txt")

    else:
        log.info(f"Read null associations from {args.null_assoc}")
        null_assoc = pd.read_csv(args.null_assoc, sep='\t', header=None, 
                                 names=["SAMPLE_ID", "GENE_ID", "INDEX", "STAAR-O"])

        null_assoc_results = summarize_null_results(
            tfce,
            null_assoc,
            args.sig_thresh
        )

        np.savetxt(f"{args.out}.txt", null_assoc_results)
        log.info(f"\nSaved TFCE of null associations to {args.out}.txt")
