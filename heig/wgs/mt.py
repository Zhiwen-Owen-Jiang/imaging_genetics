import shutil
import numpy as np
import pandas as pd
import hail as hl
from hail.linalg import BlockMatrix
from scipy.sparse import coo_matrix, save_npz, load_npz
from heig.wgs.utils import *

"""
TODO:
1. Merge multiple sparse genotype datasets

"""

def check_input(args, log):
    if args.bfile is None and args.vcf is None and args.geno_mt is None:
        raise ValueError("--bfile or --vcf or --geno-mt is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    
    if args.geno_mt is not None:
        args.vcf, args.bfile = None, None
    if args.bfile is not None:
        args.vcf = None
    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")

    if args.qc_mode is None:
        args.qc_mode = 'gwas'
    log.info(f"Set QC mode as {args.qc_mode}.")
    if args.qc_mode == 'gwas' and args.save_sparse_genotype:
        raise ValueError('GWAS data cannot be saved as sparse genotype')
    if args.bfile is not None or args.vcf is not None and args.save_sparse_genotype:
        log.info(('WARNING: directly saving a bfile or vcf as a sparse genotype can be '
                  'very slow. Convert the bfile or vcf into mt first.'))


def prepare_vset(snps_mt, variant_type):
    """
    Extracting data from MatrixTable

    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype data
    variant_type: variant type

    Returns:
    ---------
    vset: (m, n) csr_matrix of genotype
    locus: a hail.Table of locus info

    """
    locus = snps_mt.rows().key_by().select('locus', 'alleles')
    locus = locus.annotate_globals(reference_genome=locus.locus.dtype.reference_genome.name)
    locus = locus.annotate_globals(variant_type=variant_type)
    bm = BlockMatrix.from_entry_expr(
        snps_mt.flipped_n_alt_alleles, mean_impute=True
    )
    if bm.shape[0] == 0 or bm.shape[1] == 0:
        raise ValueError("no variant in the genotype data")
    
    entries = bm.entries()
    non_zero_entries = entries.filter(entries.entry > 0)
    non_zero_entries = non_zero_entries.collect()
    rows = [entry['i'] for entry in non_zero_entries]
    cols = [entry['j'] for entry in non_zero_entries]
    values = np.array([entry['entry'] for entry in non_zero_entries], dtype=np.float16)

    vset = coo_matrix((values, (rows, cols)), shape=bm.shape, dtype=np.float16)
    vset = vset.tocsr()

    return vset, locus


class SparseGenotype:
    """
    This module is used in --rv-sumstats
    order of steps:
    1. keep(), update maf
    2. extract_exclude_locus() and extract_chr_interval()
    3. extract_maf()

    """
    def __init__(self, prefix, mac_thresh):
        """"
        vset (m, n): csr_matrix
        locus: a hail.Table of locus info
        ids: a pd.DataFrame of ids with index FID and IID
        
        """
        self.vset = load_npz(f"{prefix}_genotype.npz")
        self.locus = hl.read_table(f"{prefix}_locus_info.ht").key_by("locus", "alleles")
        self.ids = pd.read_csv(f"{prefix}_id.txt", sep='\t', header=None, dtype={0: object, 1: object})
        self.ids = self.ids.rename({0: "FID", 1: "IID"}, axis=1)
        self.mac_thresh = mac_thresh

        self.locus = self.locus.add_index('idx')
        self.geno_ref = self.locus.reference_genome.collect()[0]
        self.ids['idx'] = list(range(self.ids.shape[0]))
        self.ids = self.ids.set_index(["FID", "IID"])
        self.variant_idxs = np.arange(self.vset.shape[0])
        self.maf, self.is_rare = self._update_maf()

    def extract_exclude_locus(self, extract_locus, exclude_locus):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
        exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format

        """
        if extract_locus is not None:
            extract_locus = parse_locus(extract_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(extract_locus.contains(self.locus.locus))
        if exclude_locus is not None:
            exclude_locus = parse_locus(exclude_locus["locus"], self.geno_ref)
            self.locus = self.locus.filter(~exclude_locus.contains(self.locus.locus))

    def extract_chr_interval(self, chr_interval=None):
        """
        Extacting a chr interval

        Parameters:
        ------------
        chr_interval: chr interval to extract

        """
        if chr_interval is not None:
            chr, start, end = parse_interval(chr_interval, self.geno_ref)
            interval = hl.locus_interval(chr, start, end, reference_genome=self.geno_ref)
            self.locus = self.locus.filter(interval.contains(self.locus.locus))
    
    def extract_maf(self, maf_min=None, maf_max=None):
        """
        Extracting variants by MAF
        this method will only be invoked after keep()
        
        """
        if maf_min is None:
            maf_min = 0
        if maf_max is None:
            maf_max = 0.5
        self.variant_idxs = self.variant_idxs[(self.maf > maf_min) & (self.maf <= maf_max)]

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex of FID and IID

        Returns:
        ---------
        self.id_idxs: numeric indices of subjects

        """
        if not isinstance(keep_idvs, pd.MultiIndex):
            raise TypeError('keep_idvs must be a pd.MultiIndex instance')
        # self.ids = self.ids[self.ids.index.isin(keep_idvs)]
        self.ids = self.ids.loc[keep_idvs]
        if len(self.ids) == 0:
            raise ValueError('no subject in genotype data')
        self.vset = self.vset[:, self.ids['idx'].values]
        self.maf, self.is_rare = self._update_maf()
        
    def _update_maf(self):
        mac = np.squeeze(np.array(self.vset.sum(axis=1)))
        maf = mac / (self.vset.shape[1] * 2)
        is_rare = mac <= self.mac_thresh

        return maf, is_rare
    
    def parse_data(self):
        """
        Parsing genotype data as a result of filtering
        
        """
        locus_idxs = set(self.locus.idx.collect())
        common_variant_idxs_set = locus_idxs.intersection(self.variant_idxs)
        locus = self.locus.filter(hl.literal(common_variant_idxs_set).contains(self.locus.idx))
        locus = locus.drop('idx')
        common_variant_idxs = sorted(list(common_variant_idxs_set))
        common_variant_idxs = np.array(common_variant_idxs)
        vset = self.vset[common_variant_idxs]
        maf = self.maf[common_variant_idxs]
        is_rare = self.is_rare[common_variant_idxs]
        
        return vset, locus, maf, is_rare


def run(args, log):
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # read genotype data
        gprocessor = read_genotype_data(args, log)

        # do preprocessing
        log.info(f"Processing genotype data ...")
        gprocessor.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        gprocessor.extract_exclude_snps(args.extract, args.exclude)
        gprocessor.extract_chr_interval(args.chr_interval)
        gprocessor.keep_remove_idvs(args.keep, args.remove)
        gprocessor.do_processing(mode=args.qc_mode, skip=True)

        # save
        if args.save_sparse_genotype:
            log.info("Computing sparse genotype ...")
            vset, locus = prepare_vset(gprocessor.snps_mt, args.variant_type)
            log.info(f"{vset.shape[1]} subjects and {vset.shape[0]} variants in the sparse genotype")
            snps_mt_ids = gprocessor.subject_id()
            save_npz(f"{args.out}_genotype.npz", vset)
            locus.write(f"{args.out}_locus_info.ht", overwrite=True)
            snps_mt_ids = pd.DataFrame({"FID": snps_mt_ids, "IID": snps_mt_ids})
            snps_mt_ids.to_csv(f"{args.out}_id.txt", sep='\t', header=None, index=None)
            log.info((f"Save sparse genotype data at\n"
                      f"{args.out}_genotype.npz\n"
                      f"{args.out}_locus_info.ht\n"
                      f"{args.out}_id.txt"))
        else:
            gprocessor.snps_mt.write(f"{args.out}.mt", overwrite=True)
            # post check
            gprocessor = GProcessor.read_matrix_table(f"{args.out}.mt")
            try:
                gprocessor.check_valid()
            except:
                shutil.rmtree(f"{args.out}.mt")
                raise
            log.info(f"Save genotype data at {args.out}.mt")
    finally:
        clean(args.out)