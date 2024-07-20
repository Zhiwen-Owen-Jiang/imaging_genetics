import os
import hail as hl
import heig.input.dataset as ds
from heig.wgs.utils import GProcessor 

"""
TODO: 
1. add an argument for inputing hail config with a JSON file
2. does it support genotype data preprocessing?

"""


SELECTED_ANNOT = {
    'apc_conservation': hl.tfloat32,
    'apc_epigenetics': hl.tfloat32,
    'apc_epigenetics_active': hl.tfloat32,
    'apc_epigenetics_repressed': hl.tfloat32,
    'apc_epigenetics_transcription': hl.tfloat32,
    'apc_local_nucleotide_diversity': hl.tfloat32,
    'apc_mappability': hl.tfloat32,
    'apc_protein_function': hl.tfloat32,
    'apc_transcription_factor': hl.tfloat32,
    'cage_tc': hl.tstr,
    'metasvm_pred': hl.tstr,
    'rsid': hl.tstr,
    'fathmm_xf': hl.tfloat32,
    'genecode_comprehensive_category': hl.tstr,
    'genecode_comprehensive_info': hl.tstr,
    'genecode_comprehensive_exonic_category': hl.tstr,
    'genecode_comprehensive_exonic_info': hl.tstr,
    'genehancer': hl.tstr,
    'linsight': hl.tfloat32,
    'cadd_phred': hl.tfloat32,
    'rdhs': hl.tstr
}


class Annotation:
    def __init__(self, annot, geno_ref):
        """
        Processing FAVOR functional annotations

        Parameters:
        ------------
        annot: a Table of functional annotation
        geno_ref: reference genome
        
        """
        self.annot = annot
        self.geno_ref = geno_ref
        
        self._create_keys()
        self._drop_rename()
        self._convert_datatype()
        self._add_more_annot()

    @classmethod
    def read_annot(cls, favor_db, geno_ref, *args, **kwargs):
        """
        Reading FAVOR annotation

        Parameters:
        ------------
        favor_db: directory to unzipped annotation folder
            Hail will read all annotation files
        geno_ref: reference genome
        
        """
        annot = hl.import_table(favor_db, *args, **kwargs)
        return cls(annot, geno_ref)
    
    def _create_keys(self):
        """
        Creating keys for merging
        
        """
        if self.geno_ref == 'GRCh38':
            self.annot = self.annot.annotate(chromosome=hl.str('chr') + self.annot.chromosome)
        chromosome = self.annot.chromosome
        position = hl.int(self.annot.position)
        ref_allele = self.annot.ref_vcf
        alt_allele = self.annot.alt_vcf
        locus = hl.locus(chromosome, position, reference_genome=self.geno_ref)

        self.annot = self.annot.annotate(locus=locus, alleles=[ref_allele, alt_allele])
        self.annot = self.annot.key_by('locus', 'alleles')

    def _drop_rename(self):
        """
        Dropping fields and renaming annotation names
        
        """
        self.annot = self.annot.drop('apc_conservation', 'apc_local_nucleotide_diversity')

        self.annot = self.annot.rename(
            {'apc_conservation_v2': 'apc_conservation',
             'apc_local_nucleotide_diversity_v3': 'apc_local_nucleotide_diversity',
             'apc_protein_function_v3': 'apc_protein_function'}
        )

        annot_name = list(self.annot.row_value.keys())
        self.annot = self.annot.drop(*[field for field in annot_name if field not in SELECTED_ANNOT])
    
    def _convert_datatype(self):
        """
        Converting numerical columns to float

        """
        self.annot = self.annot.annotate(
            apc_conservation = hl.float32(self.annot.apc_conservation),
            apc_epigenetics = hl.float32(self.annot.apc_epigenetics),
            apc_epigenetics_active = hl.float32(self.annot.apc_epigenetics_active),
            apc_epigenetics_repressed = hl.float32(self.annot.apc_epigenetics_repressed),
            apc_epigenetics_transcription = hl.float32(self.annot.apc_epigenetics_transcription),
            apc_local_nucleotide_diversity = hl.float32(self.annot.apc_local_nucleotide_diversity),
            apc_mappability = hl.float32(self.annot.apc_mappability),
            apc_protein_function = hl.float32(self.annot.apc_protein_function), 
            apc_transcription_factor = hl.float32(self.annot.apc_transcription_factor),
            fathmm_xf = hl.float32(self.annot.fathmm_xf),
            linsight = hl.float32(self.annot.linsight),        
            cadd_phred = hl.float32(self.annot.cadd_phred)                                                    
        )

    def _add_more_annot(self):
        """
        Filling NA for cadd_phred and creating a new annotation
        
        """
        annot_local_div = -10 * hl.log10(1 - 10 ** (-self.annot.apc_local_nucleotide_diversity/10))
        self.annot = self.annot.annotate(
            cadd_phred = hl.coalesce(self.annot.cadd_phred, 0),
            apc_local_nucleotide_diversity2 = annot_local_div
        )


def check_input(args, log):
    # required arguments
    if args.vcf is None:
        raise ValueError('--vcf is required')
    if args.favor_db is None:
        raise ValueError('--favor-db is required')
    
    # required files must exist
    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"{args.vcf} does not exist")
    if not os.path.exists(args.favor_db):
        raise FileNotFoundError(f"{args.favor_db} does not exist")
    args.favor_db = os.path.join(args.favor_db, 'chr*.csv')
    
    # process arguments
    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference.')

    return geno_ref


def run(args, log):
    # check input and init
    geno_ref = check_input(args, log)
    hl.init(quiet=True, local='local[8]', 
            driver_cores=2, driver_memory='highmem', 
            worker_cores=6, worker_memory='highmem')
    hl.default_reference = geno_ref

    # convert VCF to MatrixTable
    log.info(f'Read VCF from {args.vcf}')
    gprocessor = GProcessor.import_vcf(args.vcf, geno_ref)

    # keep idvs
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
        gprocessor.extract_idvs(keep_idvs)
        
    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} variants in --extract.")
        gprocessor.extract_snps(keep_snps)
    vcf_mt = gprocessor.snps_mt

    # read annotation and preprocess
    log.info(f'Read FAVOR annotation from {args.favor_db}')
    log.info(f'Processing annotation and annotating the VCF file ...')
    annot = Annotation.read_annot(args.favor_db, geno_ref, delimiter=',', 
                                  missing='', quote='"')
    vcf_mt = vcf_mt.annotate_rows(fa=annot.annot[vcf_mt.locus, vcf_mt.alleles])

    # save the MatrixTable
    out_dir = f'{args.out}_annotated_vcf.mt'
    vcf_mt.write(out_dir, overwrite=True)
    log.info(f'Write annotated VCF to MatrixTable {out_dir}')
