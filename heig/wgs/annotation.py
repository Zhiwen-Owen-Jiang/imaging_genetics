import hail as hl
from heig.wgs.utils import *


"""
Headers are required in annotation files
The first column must be variant string
Each variant is in `chr:pos:ref:alt` format

TODO:
1. provide arguments for annot-names and extract-chr-pos

"""

class Annotation:

    """
    Processing and saving functional annotations

    """

    def __init__(self, annot, grch37):
        """
        Parameters:
        ------------
        annot: a Table of functional annotation
        grch37: if the reference genome is GRCh37

        """
        self.annot = annot
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self._create_keys()

    @classmethod
    def read_annot(cls, annot, grch37, *args, **kwargs):
        """
        Reading annotation

        Parameters:
        ------------
        annot: directory to unzipped annotation folder
            Hail will read all annotation files
        grch37: if the reference genome is GRCh37

        """
        annot = hl.import_table(annot, *args, **kwargs)
        return cls(annot, grch37)

    def _create_keys(self):
        """
        Creating keys for merging

        """
        variant_column = self.annot.row.keys()[0]
        self.annot = self.annot.annotate(parsed_variant=hl.parse_variant(self.annot[variant_column], reference_genome=self.geno_ref))
        self.annot = self.annot.annotate(
            locus=self.annot.parsed_variant.locus,
            alleles=self.annot.parsed_variant.alleles
        )
        self.annot = self.annot.key_by("locus", "alleles")
        self.annot = self.annot.drop("parsed_variant")
        
    def extract_variants(self, variant_list):
        """
        variant_list: a list/array/series of variant strings in format: `chr:pos`,
        where both `chr` and `pos` are integers.
        Only the first variant will be checked.
        
        """
        variant_list = list(variant_list)
        if not ':' in variant_list[0]:
            raise ValueError('variant must be in `chr:pos` format')
        
        parsed_variants = [
            hl.parse_locus(v, reference_genome=self.geno_ref)
            for v in variant_list
        ]

        variant_set = hl.literal(set(parsed_variants))
        self.annot = self.annot.filter(variant_set.contains(self.annot.locus))

    def extract_by_interval(self, interval_list):
        """
        interval_list: a list/array/series of interval strings in format: `chr:start-end`,
        where `chr`, `start`, and `end` are integers, and `start` < `end`.
        
        """
        interval_list = list(interval_list)
        parsed_intervals = [
            hl.parse_locus_interval(v, reference_genome=self.geno_ref)
            for v in interval_list
        ]
        interval_expr = hl.literal(parsed_intervals)
        self.annot = self.annot.filter(hl.any(lambda interval: interval.contains(self.annot.locus), interval_expr))

    def extract_annots(self, annot_list):
        """
        annot_list: a list/array/series annotation names to extract
        
        """
        annot_list = list(annot_list)
        self.annot = self.annot.select(*annot_list)

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
    if args.annot_names is not None:
        annot.extract_annots(args.annot_names)
        log.info(f"Extracted annotations {args.annot_names}.")

    if args.range is not None:
        annot.extract_by_interval(args.range)
        log.info(f"Extracted variants in specified interval(s).")

    if args.extract_chr_pos is not None:
        annot.extract_variants(args.extract_chr_pos)
        log.info(f"Extracted variants in --extract-chr-pos.")
    
    annot.write(args.out)
    log.info(f"\nSave the annotations to {args.out}_annot.ht")