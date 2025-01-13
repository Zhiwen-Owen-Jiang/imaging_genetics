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
    "init_hail",
    "parse_interval",
    "get_temp_path",
    "parse_locus",
    "read_genotype_data",
    "format_output"
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
    
    if "spark.local.dir" not in spark_conf:
        spark_conf["spark.local.dir"] = out + '_spark'

    if grch37:
        geno_ref = "GRCh37"
    else:
        geno_ref = "GRCh38"
    log.info(f"Set {geno_ref} as the reference genome.")

    tmpdir = out + '_tmp'
    logdir = out + '_hail.log'
    hl.init(
        quiet=True, 
        spark_conf=spark_conf, 
        local_tmpdir=tmpdir, 
        log=logdir, 
        tmp_dir=tmpdir
    )
    hl.default_reference = geno_ref


class GProcessor:
    MODE = {
        "gwas": {
            "defaults": {"maf_min": 0, "maf_max": 0.5},
            # "defaults": {},
            "methods": [
                "_extract_variant_type",
                "_extract_maf",
                "_extract_call_rate",
                "_filter_hwe",
                # "_extract_chr_interval"
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
                # "_extract_chr_interval": ["chr", "start", "end"]
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
                # "_vcf_filter",
                "_flip_snps",
                "_extract_variant_type",
                "_extract_maf",
                "_extract_call_rate",
                "_filter_hwe",
                "_annotate_rare_variants",
                # "_filter_missing_alt",
                # "_impute_missing_snps"
                # "_extract_chr_interval"
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
                # "_extract_chr_interval": ["chr", "start", "end"]
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
        snps_mt: a hl.MatrixTable genotype data
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
        self.chr, self.start, self.end = None, None, None
        self.n_sub = snps_mt.count_cols()
        self.n_variants = snps_mt.count_rows()
        self.logger = logging.getLogger(__name__)
        self.logger.info((f"{self.n_sub} subjects and "
                          f"{self.n_variants} variants in the genotype data.\n"))

    def do_processing(self, mode, skip=False):
        """
        Processing genotype data in a dynamic way
        extract_idvs() should be done before it

        Parameters:
        ------------
        mode: the analysis mode which affects preprocessing pipelines
            should be one of ('gwas', 'wgs'). For a given mode, there
            are default filtering and optional filtering.
        skip: boolean, if skip '_vcf_filter' and '_filter_missing_alt' 
        """
        self.snps_mt = hl.variant_qc(self.snps_mt, name="info")
        self.snps_mt = self.snps_mt.annotate_rows(
            minor_allele_index=hl.argmin(self.snps_mt.info.AF) # Index of the minor allele
        )
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
            self.logger.info("Flipped alleles for those with a MAF > 0.5")
            if not skip:
                self.logger.info("Removed variants with missing alternative alleles.")
                self.logger.info("Extracted variants with PASS in FILTER.")
                methods += ['_vcf_filter', '_filter_missing_alt']
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
        self.n_variants = self.snps_mt.count_rows()
        self.n_sub = self.snps_mt.count_cols()
        if self.n_variants == 0:
            raise ValueError("no variant remaining after preprocessing")
        if self.n_sub == 0:
            raise ValueError("no subject remaining after preprocessing")
            
    def subject_id(self):
        """
        Extracting subject ids

        Returns:
        ---------
        snps_mt_ids: a list of subject ids

        """
        snps_mt_ids = self.snps_mt.s.collect()
        if len(snps_mt_ids) == 0:
            raise ValueError("no subject remaining in the genotype data")
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
        # if "maf" not in self.snps_mt.row:
            # self.snps_mt = self.snps_mt.annotate_rows(
            #     maf=hl.if_else(
            #         self.snps_mt.info.AF[-1] > 0.5,
            #         1 - self.snps_mt.info.AF[-1],
            #         self.snps_mt.info.AF[-1],
            #     )
            # )
        self.snps_mt = self.snps_mt.annotate_rows(
            # maf=hl.min(self.snps_mt.info.AF)
            maf=self.snps_mt.info.AF[self.snps_mt.minor_allele_index]
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
        # self.snps_mt = self.snps_mt.annotate_rows(
        #     minor_allele_index=hl.argmin(self.snps_mt.info.AF) # Index of the minor allele
        # )
        self.snps_mt = self.snps_mt.annotate_entries(
            flipped_n_alt_alleles=hl.if_else(
                self.snps_mt.info.AF[self.snps_mt.minor_allele_index] > 0.5,
                self.snps_mt.GT.ploidy - self.snps_mt.GT.n_alt_alleles(),
                self.snps_mt.GT.n_alt_alleles(),
            )
        )
        # self.snps_mt = self.snps_mt.annotate_entries(
        #     flipped_n_alt_alleles=hl.if_else(
        #         self.snps_mt.info.AF[-1] > 0.5,
        #         2 - self.snps_mt.GT.n_alt_alleles(),
        #         self.snps_mt.GT.n_alt_alleles(),
        #     )
        # )
        # self.snps_mt = self.snps_mt.annotate_rows(
        #     maf=hl.if_else(
        #         self.snps_mt.info.AF[-1] > 0.5,
        #         1 - self.snps_mt.info.AF[-1],
        #         self.snps_mt.info.AF[-1],
        #     )
        # )
        
    # def _impute_missing_snps(self):
    #     """
    #     Imputing missing SNPs after flipping alleles
        
    #     """
    #     if "flipped_n_alt_alleles" in self.snps_mt.entry:
    #         column_means = self.snps_mt.aggregate_entries(
    #             hl.agg.group_by(
    #                 self.snps_mt.col_key, 
    #                 hl.agg.mean(self.snps_mt.flipped_n_alt_alleles)
    #             )
    #         )
    #         self.snps_mt = self.snps_mt.annotate_entries(
    #             flipped_n_alt_alleles=hl.or_else(
    #                 self.snps_mt.flipped_n_alt_alleles, 
    #                 column_means[self.snps_mt.col_key]
    #             )
    #         )
    #     else:
    #         raise ValueError(f"call _flip_snps() before doing imputation")

    def _annotate_rare_variants(self):
        """
        Annotating if variants have a MAC <= mac_thresh

        """
        # self.snps_mt = self.snps_mt.annotate_rows(
        #     is_rare=hl.if_else(
        #         (
        #             (self.snps_mt.info.AC[-1] <= self.mac_thresh)
        #             | (
        #                 self.snps_mt.info.AN - self.snps_mt.info.AC[-1]
        #                 <= self.mac_thresh
        #             )
        #         ),
        #         True,
        #         False,
        #     )
        # )

        self.snps_mt = self.snps_mt.annotate_rows(
            is_rare=self.snps_mt.info.AC[self.snps_mt.minor_allele_index] <= self.mac_thresh
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

    def extract_exclude_snps(self, extract_variants, exclude_variants):
        """
        Extracting and excluding variants

        Parameters:
        ------------
        extract_variants: a pd.DataFrame of SNPs
        exclude_variants: a pd.DataFrame of SNPs

        """
        if extract_variants is not None:
            extract_variants = hl.literal(set(extract_variants["SNP"]))
            self.snps_mt = self.snps_mt.filter_rows(extract_variants.contains(self.snps_mt.rsid))

        if exclude_variants is not None:
            exclude_variants = hl.literal(set(exclude_variants["SNP"]))
            self.snps_mt = self.snps_mt.filter_rows(~exclude_variants.contains(self.snps_mt.rsid))

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
            self.snps_mt = self.snps_mt.filter_rows(extract_locus.contains(self.snps_mt.locus))

        if exclude_locus is not None:
            exclude_locus = parse_locus(exclude_locus["locus"], self.geno_ref)
            self.snps_mt = self.snps_mt.filter_rows(~exclude_locus.contains(self.snps_mt.locus))
            
    def extract_chr_interval(self, chr_interval=None):
        """
        Extracting a chr interval
        
        """
        if chr_interval is not None:
            self.chr, self.start, self.end = parse_interval(chr_interval, self.geno_ref)
            self.logger.info(f"Extracted variants in {self.chr} from {self.start} to {self.end}")
            interval = hl.locus_interval(self.chr, self.start, self.end, reference_genome=self.geno_ref)
            self.snps_mt = self.snps_mt.filter_rows(interval.contains(self.snps_mt.locus))

    def keep_remove_idvs(self, keep_idvs, remove_idvs=None):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex/list/tuple/set of subject ids
        remove_idvs: a pd.MultiIndex/list/tuple/set of subject ids

        """
        if keep_idvs is not None:
            if isinstance(keep_idvs, pd.MultiIndex):
                keep_idvs = keep_idvs.get_level_values("IID").tolist()
            keep_idvs = hl.literal(set(keep_idvs))
            self.snps_mt = self.snps_mt.filter_cols(keep_idvs.contains(self.snps_mt.s))

        if remove_idvs is not None:
            if isinstance(remove_idvs, pd.MultiIndex):
                remove_idvs = remove_idvs.get_level_values("IID").tolist()
            remove_idvs = hl.literal(set(remove_idvs))
            self.snps_mt = self.snps_mt.filter_cols(~remove_idvs.contains(self.snps_mt.s))
        
    def extract_range(self):
        """
        Obtaining the chr and the max and min position
        only containing a single chr is valid  
        
        """
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
    
    def cache(self):
        self.snps_mt = self.snps_mt.cache()
        self.logger.info("Caching the genotype data in memory.")


def read_genotype_data(args, log):
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
            mac_thresh=args.mac_thresh,
            call_rate=args.call_rate,
    )
    
    return gprocessor


def parse_interval(range, geno_ref=None):
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
                "--chr-interval (--range) should be in this format: <CHR>:<POS1>,<CHR>:<POS2>"
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
        if geno_ref == 'GRCh37':
            start_chr = str(start_chr)
        elif geno_ref == 'GRCh38':
            start_chr = 'chr' + str(start_chr)
    else:
        start_chr, start_pos, end_pos = None, None, None

    return start_chr, start_pos, end_pos


def get_temp_path(outpath):
    """
    Generating a path for saving temporary files

    """
    temp_path = outpath + "_temp"
    i = np.random.choice(1000000, 1)[0]  # randomly select a large number
    temp_path += str(i)

    return temp_path


def parse_locus(variant_list, geno_ref):
    """
    Parsing locus from a list of string
    
    """
    variant_list = list(variant_list)
    if variant_list[0].count(':') != 1:
        raise ValueError('variant must be in `chr:pos` format')
    
    parsed_variants = [
        hl.parse_locus(v, reference_genome=geno_ref)
        for v in variant_list
    ]

    variant_set = hl.literal(set(parsed_variants))

    return variant_set


def format_output(cate_pvalues, n_variants, voxels, chr, start, end, set_name):
    """
    organizing pvalues to a structured format

    Parameters:
    ------------
    cate_pvalues: a pd.DataFrame of pvalues of the variant category
    n_variants: #variants of the category
    chr: chr of the variant set
    start: start location
    end: end location
    voxels: zero-based voxel idxs of the image
    set_name: can be variant category or window index

    Returns:
    ---------
    output: a pd.DataFrame of pvalues with metadata

    """
    meta_data = pd.DataFrame(
        {
            "INDEX": voxels + 1,
            "SET_NAME": set_name,
            "CHR": chr,
            "START": start,
            "END": end,
            "N_VARIANT": n_variants,
        }
    )
    output = pd.concat([meta_data, cate_pvalues], axis=1)
    return output


# class Table:
#     """
#     hail.Table processor for loci and annotations

#     """
#     def __init__(self, table, grch37):
#         """
#         table: a hail.Table
        
#         """
#         self.table = table
#         self.geno_ref = "GRCh37" if grch37 else "GRCh38"
#         self.table = self.table.add_index('idx')
#         self._create_keys()

#     @classmethod  
#     def read_table(cls, dir, grch37):
#         """
#         Reading a hail.Table from a directory

#         """
#         table = hl.read_table(dir)
#         return cls(table, grch37)
        
#     def _create_keys(self):
#         self.table = self.table.key_by('locus')

#     def extract_locus(self, extract_locus, exclude_locus):
#         """
#         Extracting variants by locus

#         Parameters:
#         ------------
#         extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
#         exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format

#         """
#         if extract_locus is not None:
#             extract_locus = hl.Table.from_pandas(extract_locus[["locus"]])
#             extract_locus = extract_locus.annotate(locus=hl.parse_locus(extract_locus.locus))
#             # filtered_with_index = self.locus.semi_join(extract_locus)
#             filtered_with_index = self.locus.filter(extract_locus.contains(self.locus.locus))
#             indices = filtered_with_index.idx.collect()
#         if exclude_locus is not None:
#             exclude_locus = hl.Table.from_pandas(exclude_locus[["locus"]])
#             exclude_locus = exclude_locus.annotate(locus=hl.parse_locus(exclude_locus.locus))
#             filtered_with_index = self.locus.filter(~exclude_locus.contains(self.locus.locus))
#             # self.logger.info(f"{self.n_variants} variants remaining after --exclude.")

    