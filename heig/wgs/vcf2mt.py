import os
import hail as hl
import heig.input.dataset as ds
from heig.wgs.utils import GProcessor, init_hail

"""
TODO: 
1. add more preprocessing options 

"""


class Annotation:
    def __init__(self, annot, grch37):
        """
        Processing FAVOR functional annotations

        Parameters:
        ------------
        annot: a Table of functional annotation
        grch37: if the reference genome is GRCh37

        """
        self.annot = annot
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self._create_keys()

    @classmethod
    def read_annot(cls, favor_annot, grch37, *args, **kwargs):
        """
        Reading FAVOR annotation

        Parameters:
        ------------
        favor_annot: directory to unzipped annotation folder
            Hail will read all annotation files
        grch37: if the reference genome is GRCh37

        """
        annot = hl.import_table(favor_annot, *args, **kwargs)
        return cls(annot, grch37)

    def _create_keys(self):
        """
        Creating keys for merging

        """
        if self.geno_ref == "GRCh38":
            self.annot = self.annot.annotate(
                chromosome=hl.str("chr") + self.annot.chromosome
            )
            
        try:
            chromosome = self.annot.chromosome
            position = hl.int(self.annot.position)
            ref_allele = self.annot.ref_vcf
            alt_allele = self.annot.alt_vcf
        except:
            raise ValueError('chromosome, position, ref_vcf, alt_vcf are required columns')
        locus = hl.locus(chromosome, position, reference_genome=self.geno_ref)

        self.annot = self.annot.annotate(locus=locus, alleles=[ref_allele, alt_allele])
        self.annot = self.annot.key_by("locus", "alleles")
        

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
        super.__init__(annot, grch37)

        self._drop_rename()
        # self._convert_datatype()
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

    # def _convert_datatype(self):
    #     """
    #     Converting numerical columns to float

    #     """
    #     self.annot = self.annot.annotate(
    #         apc_conservation=hl.float32(self.annot.apc_conservation),
    #         apc_epigenetics=hl.float32(self.annot.apc_epigenetics),
    #         apc_epigenetics_active=hl.float32(self.annot.apc_epigenetics_active),
    #         apc_epigenetics_repressed=hl.float32(self.annot.apc_epigenetics_repressed),
    #         apc_epigenetics_transcription=hl.float32(
    #             self.annot.apc_epigenetics_transcription
    #         ),
    #         apc_local_nucleotide_diversity=hl.float32(
    #             self.annot.apc_local_nucleotide_diversity
    #         ),
    #         apc_mappability=hl.float32(self.annot.apc_mappability),
    #         apc_protein_function=hl.float32(self.annot.apc_protein_function),
    #         apc_transcription_factor=hl.float32(self.annot.apc_transcription_factor),
    #         fathmm_xf=hl.float32(self.annot.fathmm_xf),
    #         linsight=hl.float32(self.annot.linsight),
    #         cadd_phred=hl.float32(self.annot.cadd_phred),
    #     )

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
            cadd_phred=hl.coalesce(self.annot.cadd_phred, 0),
            apc_local_nucleotide_diversity2=annot_local_div,
        )


def check_input(args, log):
    # required arguments
    if args.vcf is None and args.bfile is None:
        raise ValueError("either --vcf or --bfile is required")
    if args.favor_annot is None and args.general_annot is None:
        raise ValueError("either --favor-annot or --general-annot is required")
    elif args.favor_annot is not None and args.general_annot is not None:
        log.info("WARNING: ignore --general-annot as --favor-annot specified.")

    # required files must exist
    ds.check_existence(args.vcf)
    ds.check_existence(args.favor_annot)
    ds.check_existence(args.general_annot)
    # args.favor_annot = os.path.join(args.favor_annot, "chr*.csv")


def run(args, log):
    # check input and init
    check_input(args, log)
    init_hail(args.spark_conf, args.grch37, log)

    # convert VCF to MatrixTable
    if args.bfile is not None:
        log.info(f"Read bfile from {args.bfile}")
        gprocessor = GProcessor.import_plink(
            args.bfile,
            grch37=args.grch37,
            variant_type=args.variant_type,
            hwe=args.hwe,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            call_rate=args.call_rate,
        )
    elif args.vcf is not None:
        log.info(f"Read VCF from {args.vcf}")
        gprocessor = GProcessor.import_vcf(
            args.vcf,
            grch37=args.grch37,
            variant_type=args.variant_type,
            hwe=args.hwe,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            call_rate=args.call_rate,
        )
    gprocessor.extract_idvs(args.keep)
    gprocessor.extract_snps(args.extract)
    gprocessor.do_processing(mode="gwas")
    vcf_mt = gprocessor.snps_mt

    # read annotation and preprocess
    if args.favor_annot is not None:
        log.info(f"Read FAVOR annotation from {args.favor_annot}")
        annot = AnnotationFAVOR.read_annot(
            args.favor_annot, args.grch37, delimiter=",", missing="", quote='"'
        )
    else:
        log.info(f"Read annotation from {args.general_annot}")
        annot = Annotation.read_annot(
            args.general_annot, args.grch37, delimiter=",", missing="", quote='"'
        )

    log.info(f"Processing annotation and annotating the genotype file ...")
    vcf_mt = vcf_mt.annotate_rows(fa=annot.annot[vcf_mt.locus, vcf_mt.alleles])

    # save the MatrixTable
    out_dir = f"{args.out}_annotated_genotype.mt"
    vcf_mt.write(out_dir, overwrite=True)
    log.info(f"\nSave the annotated genotype to MatrixTable {out_dir}")
