import logging
import hail as hl
from heig.wgs.utils import *


"""
Headers are required in annotation files
The first column must be variant string
Each variant is in `chr:pos:ref:alt` format

TODO:
1. provide arguments for annot-names and extract-chr-pos
2. why is it `interval-list`?

"""

class Annotation:
    """
    Processing and saving functional annotations

    """

    def __init__(self, annot, grch37):
        """
        Parameters:
        ------------
        annot: a hail.Table of functional annotation
        grch37: if the reference genome is GRCh37

        """
        self.annot = annot
        self.n_variants = annot.count()
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self._parse_variant()
        self._create_keys()
        self.logger = logging.getLogger(__name__)

    @classmethod
    def read_annot(cls, annot_dir, grch37, *args, **kwargs):
        """
        Reading annotation

        Parameters:
        ------------
        annot_dir: directory to unzipped annotation folder
            Hail will read all annotation files
        grch37: if the reference genome is GRCh37

        """
        annot = hl.import_table(annot_dir, *args, **kwargs)
        return cls(annot, grch37)

    def _parse_variant(self):
        """
        Parsing variants
        
        """
        variant_column = self.annot.row.keys()[0]
        parsed_variant = hl.parse_variant(self.annot[variant_column], reference_genome=self.geno_ref)
        self.annot = self.annot.annotate(parsed_variant=parsed_variant)

    def _create_keys(self):
        """
        Creating keys for merging

        """
        self.annot = self.annot.annotate(
            locus=self.annot.parsed_variant.locus,
            alleles=self.annot.parsed_variant.alleles
        )
        self.annot = self.annot.key_by("locus", "alleles")
        self.annot = self.annot.drop("parsed_variant")
        
    def extract_exclude_locus(self, extract_locus=None, exclude_locus=None):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
        exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format
        
        """
        if extract_locus is not None:
            extract_locus = parse_locus(extract_locus["locus"], self.geno_ref)
            self.annot = self.annot.filter(extract_locus.contains(self.annot.locus))
            self.n_variants = self.annot.count()
            self.logger.info(f"{self.n_variants} variants remaining after --extract-locus.")
        if exclude_locus is not None:
            exclude_locus = parse_locus(exclude_locus["locus"], self.geno_ref)
            self.annot = self.annot.filter(~exclude_locus.contains(self.annot.locus))
            self.n_variants = self.annot.count()
            self.logger.info(f"{self.n_variants} variants remaining after --exclude-locus.")

    def extract_by_interval(self, interval_list=None):
        """
        interval_list: a list/array/series of interval strings in format: `chr:start-end`,
        where `chr`, `start`, and `end` are integers, and `start` < `end`.
        
        """
        if interval_list is not None:
            interval_list = list(interval_list)
            parsed_intervals = [
                hl.parse_locus_interval(v, reference_genome=self.geno_ref)
                for v in interval_list
            ]
            interval_expr = hl.literal(parsed_intervals)
            self.annot = self.annot.filter(hl.any(lambda interval: interval.contains(self.annot.locus), interval_expr))
            self.n_variants = self.annot.count()
            self.logger.info(f"{self.n_variants} variants remaining in --chr-interval.")

    def extract_annots(self, annot_list=None):
        """
        annot_list: a list/array/series annotation names to extract
        
        """
        if annot_list is not None:
            annot_list = list(annot_list)
            self.annot = self.annot.select(*annot_list)
            self.logger.info(f"{annot_list} extracted from the annotation.")

    def save(self, output_dir):
        self.annot.write(f'{output_dir}_annot.ht', overwrite=True)
        

class AnnotationFAVOR(Annotation):

    SELECTED_ANNOT = {
        "apc_conservation": hl.tfloat32,
        "apc_epigenetics": hl.tfloat32,
        "apc_epigenetics_active": hl.tfloat32,
        "apc_epigenetics_repressed": hl.tfloat32,
        "apc_epigenetics_transcription": hl.tfloat32,
        "apc_local_nucleotide_diversity": hl.tfloat32,
        "apc_mappability": hl.tfloat32,
        "apc_protein_function": hl.tfloat32,
        "apc_transcription_factor": hl.tfloat32,
        "cage_tc": hl.tstr,
        "metasvm_pred": hl.tstr,
        "rsid": hl.tstr,
        "fathmm_xf": hl.tfloat32,
        "genecode_comprehensive_category": hl.tstr,
        "genecode_comprehensive_info": hl.tstr,
        "genecode_comprehensive_exonic_category": hl.tstr,
        "genecode_comprehensive_exonic_info": hl.tstr,
        "genehancer": hl.tstr,
        "linsight": hl.tfloat32,
        "cadd_phred": hl.tfloat32,
        "rdhs": hl.tstr,
    }

    def __init__(self, annot, grch37):
        super().__init__(annot, grch37)
        self._drop_rename()
        self._add_more_annot()

    def _parse_variant(self):
        parse_variant = hl.variant_str(hl.locus(self.annot.chromosome, self.annot.position), 
                                       self.annot.ref_vcf, self.annot.alt_vcf)
        self.annot = self.annot.annotate(parsed_variant=parse_variant)

    def _drop_rename(self):
        """
        Dropping fields and renaming annotation names

        """
        self.annot = self.annot.drop(
            "apc_conservation", "apc_local_nucleotide_diversity"
        )

        self.annot = self.annot.rename(
            {
                "apc_conservation_v2": "apc_conservation",
                "apc_local_nucleotide_diversity_v3": "apc_local_nucleotide_diversity",
                "apc_protein_function_v3": "apc_protein_function",
            }
        )

        annot_name = list(self.annot.row_value.keys())
        self.annot = self.annot.drop(
            *[field for field in annot_name if field not in self.SELECTED_ANNOT]
        )

    def _add_more_annot(self):
        """
        Filling NA for cadd_phred and creating a new annotation

        """
        self.annot = self.annot.annotate(apc_local_nucleotide_diversity=hl.float32(
                self.annot.apc_local_nucleotide_diversity
            )
        )
        annot_local_div = -10 * hl.log10(
            1 - 10 ** (-self.annot.apc_local_nucleotide_diversity / 10)
        )
        self.annot = self.annot.annotate(
            cadd_phred=hl.coalesce(self.annot.cadd_phred, '0'),
            apc_local_nucleotide_diversity2=annot_local_div,
        )


def check_input(args, log):
    # required arguments
    if args.favor_annot is None and args.general_annot is None:
        raise ValueError("either --favor-annot or --general-annot is required")
    elif args.favor_annot is not None and args.general_annot is not None:
        log.info("WARNING: ignore --general-annot as --favor-annot specified.")


def run(args, log):
    # check input and init
    check_input(args, log)
    init_hail(args.spark_conf, args.grch37, args.out, log)

    # read annotation and preprocess
    if args.favor_annot is not None:
        log.info(f"Read FAVOR annotations from {args.favor_annot}")
        annot = AnnotationFAVOR.read_annot(
            args.favor_annot, args.grch37, delimiter=",", missing="", quote='"'
        )
    else:
        log.info(f"Read annotations from {args.general_annot}")
        annot = Annotation.read_annot(
            args.general_annot, args.grch37, delimiter=",", missing="", quote='"'
        )

    log.info(f"Processing annotations ...")
    annot.extract_annots(args.annot_names)
    annot.extract_by_interval(args.chr_interval)
    annot.extract_exclude_locus(args.extract_locus, args.exclude_locus)
    
    annot.write(args.out)
    log.info(f"\nSave the annotations to {args.out}_annot.ht")