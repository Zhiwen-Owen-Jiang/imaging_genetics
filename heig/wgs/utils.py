import hail as hl
import numpy as np
import pandas as pd
import logging


__all__ = ['Annotation_name_catalog', 'Annotation_catalog_name',
           'Annotation_name', 'GProcessor', 'keep_ldrs',
           'remove_dependent_columns', 'extract_align_subjects',
           'get_common_ids']


Annotation_name_catalog = {
    'rs_num': 'rsid',
    'GENCODE.Category': 'genecode_comprehensive_category',
    'GENCODE.Info': 'genecode_comprehensive_info',
    'GENCODE.EXONIC.Category': 'genecode_comprehensive_exonic_category',
    'MetaSVM': 'metasvm_pred',
    'GeneHancer': 'genehancer',
    'CAGE': 'cage_tc',
    'DHS': 'rdhs',
    'CADD': 'cadd_phred',
    'LINSIGHT': 'linsight',
    'FATHMM.XF': 'fathmm_xf',
    'aPC.EpigeneticActive': 'apc_epigenetics_active',
    'aPC.EpigeneticRepressed': 'apc_epigenetics_repressed',
    'aPC.EpigeneticTranscription': 'apc_epigenetics_transcription',
    'aPC.Conservation': 'apc_conservation',
    'aPC.LocalDiversity': 'apc_local_nucleotide_diversity',
    'aPC.LocalDiversity(-)': 'apc_local_nucleotide_diversity2',
    'aPC.Mappability': 'apc_mappability',
    'aPC.TF': 'apc_transcription_factor',
    'aPC.Protein': 'apc_protein_function'
}


Annotation_catalog_name = dict()
for k, v in Annotation_name_catalog.items():
    Annotation_catalog_name[v] = k


Annotation_name = ["CADD",
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
                   "aPC.Protein"
                   ]


class GProcessor:
    def __init__(self, snps_mt): 
        """
        Genetic data processor

        Parameters:
        ------------
        snps_mt: a hl.MatrixTable of annotated VCF
        
        """
        self.snps_mt = snps_mt
    
    def do_processing(self, geno_ref, variant_type, keep_snps, keep_idvs, *args,
                      maf_min=None, maf_max=0.01, mac_thresh=10, **kwargs):
        """
        Processing genetic data

        Parameters:
        ------------
        variant_type: one of ('variant', 'snv', 'indel')
        geno_ref: reference genome
        keep_snps: a pd.DataFrame of SNPs
        keep_idvs: a set of subject ids
        maf_max: a float number between 0 and 0.5
        maf_min: a float number between 0 and 0.5, must be smaller than maf_max
            (maf_min, maf_max) is the maf range for analysis
        mac_thresh: a int number greater than 0, variants with a mac less than this
            will be identified as a rarer variants in ACAT-V
        
        """
        if 'fa' not in self.snps_mt.row:
            raise ValueError('--geno-mt must be annotated before doing analysis')
        
        self.variant_type = variant_type
        self.geno_ref = geno_ref
        self.maf_min = maf_min
        self.maf_max = maf_max
        self.mac_thresh = mac_thresh
        self.logger = logging.getLogger(__name__)

        if keep_snps is not None:
            self.extract_snps(keep_snps)
        if keep_idvs is not None:
            self.extract_idvs(keep_idvs)

        self.snps_mt = hl.variant_qc(self.snps_mt, name='info')
        if 'filters' in self.snps_mt.row:
            self.snps_mt = self.snps_mt.filter_rows((hl.len(self.snps_mt.filters) == 0) | 
                                                    hl.is_missing(self.snps_mt.filters))
        self._extract_variant_type()
        self._extract_maf()
        self._flip_snps()
        self._annotate_rare_variants()
        if args or kwargs:
            self._extract_gene(*args, **kwargs)

    @classmethod
    def read_data(cls, dir):
        """
        Reading data from a directory

        Parameters:
        ------------
        dir: directory to annotated VCF in MatrixTable
        
        """
        cls.logger = logging.getLogger(__name__)
        snps_mt = hl.read_matrix_table(dir)
        cls.logger.info(f"{snps_mt.count_cols()} subjects and {snps_mt.rows().count()} variants in --geno-mt.")
        
        return cls(snps_mt)

    def save_interim_data(self, temp_dir):
        self.snps_mt.write(temp_dir) # slow but fair
        self.snps_mt = hl.read_matrix_table(temp_dir)

    def check_valid(self):
        n_variants = self.snps_mt.rows().count()
        if n_variants == 0:
            raise ValueError('no variant remaining after preprocessing')
        else:
            self.logger.info(f"{n_variants} variants included in analysis.")

    def extract_subject_id(self):
        """
        Extracting subject ids

        Returns:
        ---------
        snps_mt_ids: a list of subject ids
        
        """
        snps_mt_ids = self.snps_mt.s.collect()
        return snps_mt_ids

    def _extract_variant_type(self):
        """
        Extracting variants with specified type

        """
        if self.variant_type == 'variant':
            return
        elif self.variant_type == 'snv':
            func = hl.is_snp
        elif self.variant_type == 'indel':
            func = hl.is_indel
        else:
            raise ValueError('variant_type must be snv, indel or variant')
        self.snps_mt = self.snps_mt.annotate_rows(target_type=func(self.snps_mt.alleles[0], 
                                                                   self.snps_mt.alleles[1]))
        self.snps_mt = self.snps_mt.filter_rows(self.snps_mt.target_type)

    def _extract_maf(self):
        """
        Extracting variants with a MAF < maf_max
        
        """
        if self.maf_min is None:
            self.maf_min = 0
        if self.maf_min >= self.maf_max:
            raise ValueError('maf_min is greater than maf_max')
        if 'maf' not in self.snps_mt.row:
            self.snps_mt = self.snps_mt.annotate_rows(
                maf=hl.if_else(
                    self.snps_mt.info.AF[-1] > 0.5,
                    1 - self.snps_mt.info.AF[-1],
                    self.snps_mt.info.AF[-1]
                )
            )
        self.snps_mt = self.snps_mt.filter_rows((self.snps_mt.maf >= self.maf_min) & 
                                                (self.snps_mt.maf <= self.maf_max))

    def _flip_snps(self):
        """
        Flipping variants with MAF > 0.5, and creating an annotation for maf

        """
        self.snps_mt = self.snps_mt.annotate_entries(
            flipped_n_alt_alleles=hl.if_else(
                self.snps_mt.info.AF[-1] > 0.5,
                2 - self.snps_mt.GT.n_alt_alleles(),
                self.snps_mt.GT.n_alt_alleles()
            )
        )   
        self.snps_mt = self.snps_mt.annotate_rows(
            maf=hl.if_else(
                self.snps_mt.info.AF[-1] > 0.5,
                1 - self.snps_mt.info.AF[-1],
                self.snps_mt.info.AF[-1]
            )
        ) 

    def _annotate_rare_variants(self):
        """
        Annotating if variants have a MAC < mac_thresh
        
        """
        self.snps_mt = self.snps_mt.annotate_rows(
            is_rare=hl.if_else(((self.snps_mt.info.AC[-1] < self.mac_thresh) | 
                                (self.snps_mt.info.AN - self.snps_mt.info.AC[-1] < self.mac_thresh)),
                                True, False)
        )

    def _extract_gene(self, chr, start, end, gene_name=None):
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
        chr = str(chr)
        if self.geno_ref == 'GRCh38':
            chr = 'chr' + chr
            
        if gene_name is None:
            self.snps_mt = self.snps_mt.filter_rows((self.snps_mt.locus.contig == chr) & 
                                                    (self.snps_mt.locus.position >= start) & 
                                                    (self.snps_mt.locus.position <= end))
        else:
            gencode_info = self.snps_mt.fa[Annotation_name_catalog['GENCODE.Info']]
            self.snps_mt = self.snps_mt.filter_rows(gencode_info.contains(gene_name))

    def extract_snps(self, keep_snps):
        """
        Extracting variants

        Parameters:
        ------------
        keep_snps: a pd.DataFrame of SNPs
        
        """
        keep_snps = hl.literal(set(keep_snps['SNP']))
        self.snps_mt = self.snps_mt.filter_rows(keep_snps.contains(self.snps_mt.rsid))

    def extract_idvs(self, keep_idvs):
        """
        Extracting subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex/list/tuple/set of subject ids
        
        """
        if isinstance(keep_idvs, pd.MultiIndex):
            keep_idvs = keep_idvs.get_level_values('IID').tolist()[1:] # remove 'IID'
        keep_idvs = hl.literal(set(keep_idvs))
        self.snps_mt = self.snps_mt.filter_cols(keep_idvs.contains(self.snps_mt.s))


def get_common_ids(ids, snps_mt_ids, keep_idvs=None):
    """
    Extracting common ids

    Parameters:
    ------------
    ids: a np.array of id
    snps_mt_ids: a list of id
    keep_idvs: a pd.MultiIndex of id

    Returns:
    ---------
    common_ids: a set of common ids
    
    """
    if keep_idvs is not None:
        keep_idvs = keep_idvs.get_level_values('IID').tolist()[1:]
        common_ids = set(keep_idvs).intersection(ids)
    else:
        common_ids = set(ids)
    common_ids = common_ids.intersection(snps_mt_ids)
    return common_ids


def keep_ldrs(n_ldrs, bases, resid_ldr):
    """
    Keeping top LDRs

    Parameters:
    ------------
    n_ldrs: a int number
    bases: functional bases (N, N)
    resid_ldr: LDR residuals (n, r)

    Returns:
    ---------
    bases: functional bases (N, n_ldrs)
    resid_ldr: LDR residuals (n, n_ldrs)
    
    """
    if bases.shape[1] < n_ldrs:
        raise ValueError('the number of bases is less than --n-ldrs')
    if resid_ldr.shape[1] < n_ldrs:
        raise ValueError('LDR residuals are less than --n-ldrs')
    bases = bases[:, :n_ldrs]
    resid_ldr = resid_ldr[:, :n_ldrs]
    return bases, resid_ldr


def remove_dependent_columns(matrix):
    """
    Removing dependent columns from covariate matrix

    Parameters:
    ------------
    matrix: covariate matrix including the intercept

    Returns:
    ---------
    matrix: covariate matrix w/ or w/o columns removed
    
    """
    rank = np.linalg.matrix_rank(matrix)
    if rank < matrix.shape[1]:
        _, R = np.linalg.qr(matrix)
        independent_columns = np.where(np.abs(np.diag(R)) > 1e-10)[0]
        matrix = matrix[:, independent_columns]
    return matrix


def extract_align_subjects(current_id, target_id):
    """
    Extracting and aligning subjects for a dataset based on another dataset
    target_id must be the subset of current_id

    Parameters:
    ------------
    current_id: a list or np.array of ids of the current dataset
    target_id: a list or np.array of ids of the another dataset

    Returns:
    ---------
    index: a np.array of indices such that current_id[index] = target_id

    """
    if not set(target_id).issubset(current_id):
        raise ValueError('targettarget_id must be the subset of current_id')
    n_current_id = len(current_id)
    current_id = pd.DataFrame({'id': current_id, 'index': range(n_current_id)})
    target_id = pd.DataFrame({'id': target_id})
    target_id = target_id.merge(current_id, on='id')
    index = np.array(target_id['index'])
    return index