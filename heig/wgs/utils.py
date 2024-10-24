import os
import json
import logging
import hail as hl
import numpy as np
import pandas as pd



__all__ = [
    "Annotation_name_catalog",
    "Annotation_catalog_name",
    "Annotation_name",
    "GProcessor",
    "keep_ldrs",
    "init_hail",
    "process_range",
    "get_temp_path",
]


Annotation_name_catalog = {
    "rs_num": "rsid",
    "GENCODE.Category": "genecode_comprehensive_category",
    "GENCODE.Info": "genecode_comprehensive_info",
    "GENCODE.EXONIC.Category": "genecode_comprehensive_exonic_category",
    "MetaSVM": "metasvm_pred",
    "GeneHancer": "genehancer",
    "CAGE": "cage_tc",
    "DHS": "rdhs",
    "CADD": "cadd_phred",
    "LINSIGHT": "linsight",
    "FATHMM.XF": "fathmm_xf",
    "aPC.EpigeneticActive": "apc_epigenetics_active",
    "aPC.EpigeneticRepressed": "apc_epigenetics_repressed",
    "aPC.EpigeneticTranscription": "apc_epigenetics_transcription",
    "aPC.Conservation": "apc_conservation",
    "aPC.LocalDiversity": "apc_local_nucleotide_diversity",
    "aPC.LocalDiversity(-)": "apc_local_nucleotide_diversity2",
    "aPC.Mappability": "apc_mappability",
    "aPC.TF": "apc_transcription_factor",
    "aPC.Protein": "apc_protein_function",
}


Annotation_catalog_name = dict()
for k, v in Annotation_name_catalog.items():
    Annotation_catalog_name[v] = k


Annotation_name = [
    "CADD",
    "LINSIGHT",
    "FATHMM.XF",
    "aPC.EpigeneticActive",
    "aPC.EpigeneticRepressed",
    "aPC.EpigeneticTranscription",
    "aPC.Conservation",
    "aPC.LocalDiversity",
    "aPC.LocalDiversity(-)",
    "aPC.Mappability",
    "aPC.TF",
    "aPC.Protein",
]


def init_hail(spark_conf_file, grch37, out, log):
    """
    Initializing hail

    Parameters:
    ------------
    spark_conf_file: spark configuration in json format
    grch37: if the reference genome is GRCh37
    out: output directory
    log: a logger

    """
    with open(spark_conf_file, "r") as file:
        spark_conf = json.load(file)

    if grch37:
        geno_ref = "GRCh37"
    else:
        geno_ref = "GRCh38"
    log.info(f"Set {geno_ref} as the reference genome.")

    tmpdir = out + 'tmp'
    hl.init(quiet=True, spark_conf=spark_conf, local_tmpdir=tmpdir)
    hl.default_reference = geno_ref


class GProcessor:
    MODE = {
        "gwas": {
            "defaults": {"maf_min": 0, "maf_max": 0.5},
            "methods": [
                "_extract_variant_type",
                "_extract_maf",
                "_extract_call_rate",
                "_filter_hwe",
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
            },
        },
        "wgs": {
            "defaults": {
                "geno_ref": "GRCh38",
                "variant_type": "snv",
                "maf_max": 0.01,
                "maf_min": 0,
                "mac_thresh": 10,
            },
            "methods": [
                "_vcf_filter",
                "_flip_snps",
                "_extract_variant_type",
                "_extract_maf",
                "_extract_call_rate",
                "_filter_hwe",
                "_annotate_rare_variants",
                "_filter_missing_alt",
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
            },
        },
    }

    PARAMETERS = {
        "variant_type": "Variant type",
        "geno_ref": "Reference genome",
        "maf_min": "Minimum MAF (>)",
        "maf_max": "Maximum MAF (<=)",
        "mac_thresh": "MAC threshold to annotate very rare variants (<=)",
        "call_rate": "Call rate (>=)",
        "hwe": "HWE p-value (>=)",
    }

    def __init__(
        self,
        snps_mt,
        grch37=None,
        variant_type=None,
        hwe=None,
        maf_min=None,
        maf_max=None,
        mac_thresh=None,
        call_rate=None,
    ):
        """
        Genetic data processor

        Parameters:
        ------------
        snps_mt: a hl.MatrixTable of annotated VCF
        variant_type: one of ('variant', 'snv', 'indel')
        grch37: if the reference genome is GRCh37
        maf_max: a float number between 0 and 0.5
        maf_min: a float number between 0 and 0.5, must be smaller than maf_max
            (maf_min, maf_max) is the maf range for analysis
        mac_thresh: a int number greater than 0, variants with a mac less than this
            will be identified as a rarer variants in ACAT-V
        call_rate: a float number between 0 and 1, 1 - genotype missingness
        hwe: a float number between 0 and 1, variants with a HWE pvalue less than
            this will be removed

        """
        self.snps_mt = snps_mt
        self.variant_type = variant_type
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self.maf_min = maf_min
        self.maf_max = maf_max
        self.mac_thresh = mac_thresh
        self.call_rate = call_rate
        self.hwe = hwe
        self.logger = logging.getLogger(__name__)
        # self.logger.info((f"{snps_mt.count_cols()} subjects and "
        #                   f"{snps_mt.rows().count()} variants in the genotype data.\n"))

    def do_processing(self, mode):
        """
        Processing genotype data in a dynamic way
        extract_idvs() should be done before it

        Parameters:
        ------------
        mode: the analysis mode which affects preprocessing pipelines
            should be one of ('gwas', 'wgs'). For a given mode, there
            are default filtering and optional filtering.
        """
        self.snps_mt = hl.variant_qc(self.snps_mt, name="info")
        config = self.MODE.get(mode, {})
        defaults = config.get("defaults", {})
        methods = config.get("methods", [])
        conditions = config.get("conditions", {})
        attributes = self.__dict__.keys()

        for attr in attributes:
            if attr in defaults:
                setattr(self, attr, getattr(self, attr) or defaults.get(attr))

        self.logger.info("Variant QC parameters")
        self.logger.info("---------------------")
        for para_k, para_v in self.PARAMETERS.items():
            if getattr(self, para_k) is not None:
                self.logger.info(f"{para_v}: {getattr(self, para_k)}")
        if mode == "wgs":
            self.logger.info("Removed variants with missing alternative alleles.")
            self.logger.info("Extracted variants with PASS in FILTER.")
            self.logger.info("Flipped alleles for those with a MAF > 0.5.")
        self.logger.info("---------------------\n")

        for method in methods:
            method_conditions = conditions.get(method, [])
            if all(getattr(self, attr) is not None for attr in method_conditions):
                getattr(self, method)()

    @classmethod
    def read_matrix_table(cls, dir, *args, **kwargs):
        """
        Reading MatrixTable from a directory

        Parameters:
        ------------
        dir: directory to annotated VCF in MatrixTable

        """
        snps_mt = hl.read_matrix_table(dir)
        return cls(snps_mt, *args, **kwargs)

    @classmethod
    def import_plink(cls, bfile, grch37, *args, **kwargs):
        """
        Importing genotype data from PLINK triplets

        Parameters:
        ------------
        dir: directory to PLINK triplets (prefix only)
        grch37: if the reference genome is GRCh37

        """
        geno_ref, recode = cls._recode(grch37)
        snps_mt = hl.import_plink(
            bed=bfile + ".bed",
            bim=bfile + ".bim",
            fam=bfile + ".fam",
            reference_genome=geno_ref,
            contig_recoding=recode,
        )
        return cls(snps_mt, grch37, *args, **kwargs)

    @classmethod
    def import_vcf(cls, dir, grch37, *args, **kwargs):
        """
        Importing a VCF file as MatrixTable

        Parameters:
        ------------
        dir: directory to VCF file
        grch37: if the reference genome is GRCh37
        geno_ref: reference genome

        """
        if dir.endswith("vcf"):
            force_bgz = False
        elif dir.endswith("vcf.gz") or dir.endswith("vcf.bgz"):
            force_bgz = True
        else:
            raise ValueError("VCF suffix is incorrect")

        geno_ref, recode = cls._recode(grch37)
        vcf_mt = hl.import_vcf(
            dir, force_bgz=force_bgz, reference_genome=geno_ref, contig_recoding=recode
        )
        return cls(vcf_mt, grch37, *args, **kwargs)

    @staticmethod
    def _recode(grch37):
        if not grch37:
            geno_ref = "GRCh38"
            recode = {f"{i}": f"chr{i}" for i in (list(range(1, 23)) + ["X", "Y"])}
        else:
            geno_ref = "GRCh37"
            recode = {f"chr{i}": f"{i}" for i in (list(range(1, 23)) + ["X", "Y"])}
        return geno_ref, recode

    def save_interim_data(self, temp_dir):
        """
        Saving interim MatrixTable,
        which is useful for wgs where I/O is expensive

        Parameters:
        ------------
        temp_dir: directory to temporarily save the MatrixTable

        """
        self.logger.info(f"Save preprocessed genotype data to {temp_dir}")
        self.snps_mt.write(temp_dir)  # slow but fair
        self.snps_mt = hl.read_matrix_table(temp_dir)

    def check_valid(self):
        """
        Checking non-zero #variants

        """
        n_variants = self.snps_mt.rows().count()
        if n_variants == 0:
            raise ValueError("no variant remaining after preprocessing")
        else:
            self.logger.info(f"{n_variants} variants included in analysis.")

    def subject_id(self):
        """
        Extracting subject ids

        Returns:
        ---------
        snps_mt_ids: a list of subject ids

        """
        snps_mt_ids = self.snps_mt.s.collect()
        return snps_mt_ids

    def annotate_cols(self, table, annot_name):
        """
        Annotating columns with values from a table
        the table is supposed to have the key 'IID'

        Parameters:
        ------------
        table: a hl.Table
        annot_name: annotation name

        """
        table = table.key_by("IID")
        annot_expr = {annot_name: table[self.snps_mt.s]}
        self.snps_mt = self.snps_mt.annotate_cols(**annot_expr)

    def _vcf_filter(self):
        """
        Extracting variants with a "PASS" in VCF FILTER

        """
        if "filters" in self.snps_mt.row:
            self.snps_mt = self.snps_mt.filter_rows(
                (hl.len(self.snps_mt.filters) == 0)
                | hl.is_missing(self.snps_mt.filters)
            )

    def _extract_variant_type(self):
        """
        Extracting variants with specified type

        """
        if self.variant_type == "variant":
            return
        elif self.variant_type == "snv":
            func = hl.is_snp  # the same as isSNV()
        elif self.variant_type == "indel":
            func = hl.is_indel
        else:
            raise ValueError("variant_type must be snv, indel or variant")
        self.snps_mt = self.snps_mt.annotate_rows(
            target_type=func(self.snps_mt.alleles[0], self.snps_mt.alleles[1])
        )
        self.snps_mt = self.snps_mt.filter_rows(self.snps_mt.target_type)

    def _extract_maf(self):
        """
        Extracting variants with a maf_min < MAF <= maf_max

        """
        if self.maf_min is None:
            self.maf_min = 0
        if self.maf_min > self.maf_max:
            raise ValueError("maf_min is greater than maf_max")
        if "maf" not in self.snps_mt.row:
            self.snps_mt = self.snps_mt.annotate_rows(
                maf=hl.if_else(
                    self.snps_mt.info.AF[-1] > 0.5,
                    1 - self.snps_mt.info.AF[-1],
                    self.snps_mt.info.AF[-1],
                )
            )
        self.snps_mt = self.snps_mt.filter_rows(
            (self.snps_mt.maf > self.maf_min) & (self.snps_mt.maf <= self.maf_max)
        )

    def _extract_call_rate(self):
        """
        Extracting variants with a call rate >= call_rate

        """
        self.snps_mt = self.snps_mt.filter_rows(
            self.snps_mt.info.call_rate >= self.call_rate
        )

    def _filter_hwe(self):
        """
        Filtering variants with a HWE pvalues < hwe

        """
        self.snps_mt = self.snps_mt.filter_rows(
            self.snps_mt.info.p_value_hwe >= self.hwe
        )

    def _filter_missing_alt(self):
        """
        Filtering variants with missing alternative allele

        """
        # self.snps_mt = self.snps_mt.filter_rows(self.snps_mt.alleles[1] != '*')
        self.snps_mt = self.snps_mt.filter_rows(
            hl.is_star(self.snps_mt.alleles[0], self.snps_mt.alleles[1]), keep=False
        )

    def _flip_snps(self):
        """
        Flipping variants with MAF > 0.5, and creating an annotation for maf

        """
        self.snps_mt = self.snps_mt.annotate_entries(
            flipped_n_alt_alleles=hl.if_else(
                self.snps_mt.info.AF[-1] > 0.5,
                2 - self.snps_mt.GT.n_alt_alleles(),
                self.snps_mt.GT.n_alt_alleles(),
            )
        )
        self.snps_mt = self.snps_mt.annotate_rows(
            maf=hl.if_else(
                self.snps_mt.info.AF[-1] > 0.5,
                1 - self.snps_mt.info.AF[-1],
                self.snps_mt.info.AF[-1],
            )
        )

    def _annotate_rare_variants(self):
        """
        Annotating if variants have a MAC <= mac_thresh

        """
        self.snps_mt = self.snps_mt.annotate_rows(
            is_rare=hl.if_else(
                (
                    (self.snps_mt.info.AC[-1] <= self.mac_thresh)
                    | (
                        self.snps_mt.info.AN - self.snps_mt.info.AC[-1]
                        <= self.mac_thresh
                    )
                ),
                True,
                False,
            )
        )

    def extract_unique_chrs(self):
        """
        Extracting unique chromosomes

        """
        unique_chrs = set(
            self.snps_mt.aggregate_rows(hl.agg.collect(self.snps_mt.locus.contig))
        )
        if self.geno_ref == "GRCh38":
            unique_chrs = [int(chr.replace("chr", "")) for chr in unique_chrs]
        else:
            unique_chrs = [int(chr) for chr in unique_chrs]

        return unique_chrs

    def extract_gene(self, chr, start, end, gene_name=None):
        """
        Extacting a gene with starting and end points for Coding, Slidewindow,
        for Noncoding, extracting genes from annotation

        Parameters:
        ------------
        chr: target chromosome
        start: start position
        end: end position
        gene_name: gene name, if specified, start and end will be ignored

        """
        if chr is None:
            return
        chr = str(chr)
        if self.geno_ref == "GRCh38":
            chr = "chr" + chr

        if gene_name is None:
            self.snps_mt = self.snps_mt.filter_rows(
                (self.snps_mt.locus.contig == chr)
                & (self.snps_mt.locus.position >= start)
                & (self.snps_mt.locus.position <= end)
            )
        else:
            if "fa" not in self.snps_mt.row:
                raise ValueError("--geno-mt must be annotated before doing analysis")
            gencode_info = self.snps_mt.fa[Annotation_name_catalog["GENCODE.Info"]]
            self.snps_mt = self.snps_mt.filter_rows(gencode_info.contains(gene_name))

    def extract_snps(self, keep_snps):
        """
        Extracting variants

        Parameters:
        ------------
        keep_snps: a pd.DataFrame of SNPs

        """
        if keep_snps is None:
            return
        keep_snps = hl.literal(set(keep_snps["SNP"]))
        self.snps_mt = self.snps_mt.filter_rows(keep_snps.contains(self.snps_mt.rsid))

    def extract_idvs(self, keep_idvs):
        """
        Extracting subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex/list/tuple/set of subject ids

        """
        if keep_idvs is None:
            return
        if isinstance(keep_idvs, pd.MultiIndex):
            keep_idvs = keep_idvs.get_level_values("IID").tolist()
        keep_idvs = hl.literal(set(keep_idvs))
        self.snps_mt = self.snps_mt.filter_cols(keep_idvs.contains(self.snps_mt.s))
        
    def extract_range(self):
        result = self.snps_mt.aggregate_rows(hl.struct(
            chr=hl.agg.take(self.snps_mt.locus.contig, 1)[0],
            min_pos=hl.agg.min(self.snps_mt.locus.position),
            max_pos=hl.agg.max(self.snps_mt.locus.position)
        ))

        # Save the results into variables
        chr = result.chr
        min_pos = result.min_pos
        max_pos = result.max_pos

        if self.geno_ref == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)

        return chr, min_pos, max_pos


# def get_common_ids(ids, snps_mt_ids=None, keep_idvs=None):
#     """
#     Extracting common ids

#     Parameters:
#     ------------
#     ids: a np.array of id
#     snps_mt_ids: a list of id
#     keep_idvs: a pd.MultiIndex of id

#     Returns:
#     ---------
#     common_ids: a set of common ids

#     """
#     if keep_idvs is not None:
#         keep_idvs = keep_idvs.get_level_values('IID').tolist()
#         common_ids = set(keep_idvs).intersection(ids)
#     else:
#         common_ids = set(ids)
#     if snps_mt_ids is not None:
#         common_ids = common_ids.intersection(snps_mt_ids)
#     return common_ids


def keep_ldrs(n_ldrs, resid_ldr, bases=None):
    """
    Keeping top LDRs

    Parameters:
    ------------
    n_ldrs: a int number
    resid_ldr: LDR residuals (n, r)
    bases: functional bases (N, N)

    Returns:
    ---------
    resid_ldr: LDR residuals (n, n_ldrs)
    bases: functional bases (N, n_ldrs) or None

    """
    if bases is not None:
        if bases.shape[1] < n_ldrs:
            raise ValueError("the number of bases is less than --n-ldrs")
        bases = bases[:, :n_ldrs]
    if resid_ldr.shape[1] < n_ldrs:
        raise ValueError("LDR residuals are less than --n-ldrs")
    resid_ldr = resid_ldr[:, :n_ldrs]
    return resid_ldr, bases


# def extract_align_subjects(current_id, target_id):
#     """
#     Extracting and aligning subjects for a dataset based on another dataset
#     target_id and current_id must only have order difference

#     Parameters:
#     ------------
#     current_id: a list or np.array of ids of the current dataset
#     target_id: a list or np.array of ids of the another dataset

#     Returns:
#     ---------
#     index: a np.array of indices such that current_id[index] = target_id

#     """
#     # if not set(target_id).issubset(current_id):
#     #     raise ValueError('target_id must be the subset of current_id')
#     if set(current_id) != set(target_id):
#         raise ValueError(('subjects in LDRs and covariates must be included in genetic data. '
#                           'Use --keep in when fitting the null model'))
#     n_current_id = len(current_id)
#     current_id = pd.DataFrame({'id': current_id, 'index': range(n_current_id)})
#     target_id = pd.DataFrame({'id': target_id})
#     target_id = target_id.merge(current_id, on='id')
#     index = np.array(target_id['index'])
#     return index


def pandas_to_table(df, dir):
    """
    Converting a pd.DataFrame to hail.Table

    Parameters:
    ------------
    df: a pd.DataFrame to convert, it must have a single index 'IID'
    target_id: a list or np.array of ids of the another dataset

    Returns:
    ---------
    index: a np.array of indices such that current_id[index] = target_id

    """
    if not df.index.name == "IID":
        raise ValueError("the DataFrame must have a single index IID")
    df.to_csv(f"{dir}.txt", sep="\t", na_rep="NA")

    table = hl.import_table(
        f"{dir}.txt", key="IID", impute=True, types={"IID": hl.tstr}, missing="NA"
    )

    return table


def process_range(range):
    """
    Converting range from string to readable format

    """
    if range is not None:
        try:
            start, end = range.split(",")
            start_chr, start_pos = [int(x) for x in start.split(":")]
            end_chr, end_pos = [int(x) for x in end.split(":")]
        except:
            raise ValueError(
                "--range should be in this format: <CHR>:<POS1>,<CHR>:<POS2>"
            )
        if start_chr != end_chr:
            raise ValueError(
                (
                    f"starting with chromosome {start_chr} "
                    f"while ending with chromosome {end_chr} "
                    "is not allowed"
                )
            )
        if start_pos > end_pos:
            raise ValueError(
                (
                    f"starting with {start_pos} "
                    f"while ending with position is {end_pos} "
                    "is not allowed"
                )
            )
    else:
        start_chr, start_pos, end_pos = None, None, None

    return start_chr, start_pos, end_pos


def get_temp_path():
    """
    Generating a path for saving temporary files

    """
    temp_path = "temp"
    i = np.random.choice(1000000, 1)[0]  # randomly select a large number
    temp_path += str(i)

    return temp_path
