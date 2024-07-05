import hail as hl
import shutil
import h5py
import numpy as np
import pandas as pd
from heig.wgs.staar import VariantSetTest, cauchy_combination
import heig.input.dataset as ds
from heig.wgs.utils import *

"""
Can do multiple genes in parallel

1. Covariates, ldrs, and genotypes need to merge and align
2. Assume varints have been filtered (SNV, indel)
3. the gene has been selected
4. NAs in gene has been filled and every varint is in minor allele

"""


class Coding:
    def __init__(self, snps_mt, variant_type, use_annotation_weights=True):
        """
        Extracting coding variants, generate annotation, and get index for each category

        Parameters:
        ------------
        snps_mt: a hail.MatrixTable of genotype data with annotation attached
        for a specific gene and variant type
        variant_type: one of ('variant', 'snv', 'indel')
        use_annotation_weights: if using annotation weights
        
        """
        gencode_exonic_category = snps_mt.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        gencode_category = snps_mt.fa[Annotation_name_catalog['GENCODE.Category']]
        lof_in_coding_snps_mt = ((gencode_exonic_category in {'stopgain', 'stoploss', 'nonsynonymous SNV', 'synonymous SNV'}) |
                              (gencode_category in {'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'}))
        self.snps_mt = snps_mt.filter_rows(lof_in_coding_snps_mt)
        self.gencode_exonic_category = snps_mt.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        self.gencode_category = snps_mt.fa[Annotation_name_catalog['GENCODE.Category']]
        self.metasvm_pred = snps_mt.fa[Annotation_name_catalog['MetaSVM']]
        if variant_type == 'snv' and use_annotation_weights:
            self.anno_pred, self.anno_name = self.get_annotation()
        else:
            self.anno_pred, self.anno_name = None, None
        self.category_dict = self.get_category(variant_type)

    def get_annotation(self):
        """
        Extracting and processing annotations 
       
        Returns:
        ---------
        anno_phred: a hail.Table of annotations for all coding variants

        """
        anno_cols = [Annotation_name_catalog[anno_name] for anno_name in Annotation_name]
        anno_phred = self.snps_mt.fa.select(*anno_cols)
        anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
        anno_local_div = -10 * hl.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
        anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div) 
        anno_name = Annotation_name.append('aPC.LocalDiversity(-)')
        return anno_phred, anno_name

    def get_category(self, variant_type):
        category_dict = dict()
        category_dict['plof'] = (((self.gencode_exonic_category in {'stopgain', 'stoploss'}) |
                                 (self.gencode_category in {'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'})))
        category_dict['synonymous'] = self.gencode_exonic_category == 'synonymous SNV'
        category_dict['missense'] = self.gencode_exonic_category == 'nonsynonymous SNV'
        category_dict['disruptive_missense'] = category_dict['missense'] & (self.metasvm_pred == 'D')
        category_dict['plof_ds'] = category_dict['plof'] | category_dict['disruptive_missense']

        ptv_snv = ((self.gencode_exonic_category in {'stopgain', 'stoploss'}) | 
                      (self.gencode_category in {'splicing', 'exonic;splicing'}) )
        ptv_indel = (self.gencode_exonic_category in {'frameshift deletion', 'frameshift insertion'})
        if variant_type == 'snv':
            category_dict['ptv'] = ptv_snv
            category_dict['ptv_ds'] = ptv_snv | category_dict['disruptive_missense']
        elif variant_type == 'indel':
            category_dict['ptv'] = ptv_indel
            category_dict['ptv_ds'] = ptv_indel | category_dict['disruptive_missense']
        else:
            category_dict['ptv'] = ptv_snv | ptv_indel 
            category_dict['ptv_ds'] = category_dict['ptv'] | category_dict['disruptive_missense']

        return category_dict


def single_gene_analysis(snps_mt, variant_type, vset_test,
                         variant_category, use_annotation_weights,
                         temp_path, log):
    """
    Single gene analysis

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated vcf
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
    one of ('all', 'plof', 'plof_ds', 'missense', 'disruptive_missense',
    'synonymous', 'ptv', 'ptv_ds')
    use_annotation_weights: if using annotation weights
    temp_path: a temp directory to save preprocessed MatrixTable to speed up I/O
    log: a logger
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # getting annotations and specific categories of variants    
    coding = Coding(snps_mt, variant_type, use_annotation_weights)

    # individual analysis
    snps_mt.write(temp_path, overwrite=True)
    snps_mt = hl.read_matrix_table(temp_path)
    cate_pvalues = dict()
    for cate, idx in coding.category_dict.items():
        if variant_category != 'all' and variant_category != cate: 
            cate_pvalues[cate] = None
        else:
            if coding.anno_phred is not None:
                phred_cate = np.array(coding.anno_phred[idx].collect())
            else:
                phred_cate = None
            snps_mt_cate = snps_mt.filter_rows(idx)
            vset_test.input_vset(snps_mt_cate, phred_cate)
            pvalues = vset_test.do_inference()
            cate_pvalues[cate] = {'n_variants': vset_test.n_variants, 'pvalues': pvalues}
    if 'missense' in cate_pvalues:
        cate_pvalues['missense'] = process_missense(cate_pvalues['missense'], 
                                                    cate_pvalues['disruptive_missense'])
    shutil.rmtree(temp_path)
    return cate_pvalues


def process_missense(m_pvalues, dm_pvalues):
    """
    Doing Cauchy combination for missense
    
    """
    cauchy_combination()


def format_output(cate_pvalues, start, end, n_variants, n_voxels, variant_category):
    """
    organize pvalues to a structured format

    """
    meta_data = pd.DataFrame({'INDEX': range(1, n_voxels+1), 
                              'VARIANT_CATEGORY': variant_category,
                              'START': start, 'END': end,
                              'n_variants': n_variants})
    output = pd.concat([meta_data, cate_pvalues], axis=1)
    return output


def check_input(args, log):
    pass


def run(args, log):
    # checking if input is valid
    chr, start, end, keep_snps = check_input(args, log)

    # reading data for unrelated subjects
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        var = file['var'][:]
        ids = file['ids'][:]

    bases = np.load(args.bases)
    inner_ldr = np.load(args.inner_ldr)

    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None

    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs in --extract.")
    else:
        keep_snps = None

    vset_test = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    snps_mt = hl.read_matrix_table(args.avcfmt)
    snps_mt = preprocess_mt(snps_mt, variant_type=args.variant_type, 
                            maf_thresh=args.maf_thresh, mac_thresh=args.mac_thresh, 
                            chr=chr, start=start, end=end)

    # extracting common ids
    snps_mt_ids = set(snps_mt.s.collect())
    common_ids = snps_mt_ids.intersection(ids)
    snps_mt = snps_mt.filter_cols(hl.literal(common_ids).contains(snps_mt.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis
    cate_pvalues = single_gene_analysis(snps_mt, args.variant_type, vset_test,
                               args.variant_category, args.use_annotation_weights,
                               args.temp_path, log)
    
    # format output
    n_voxels = bases.shape[0]
    for cate, cate_results in cate_pvalues.items():
        if isinstance(cate_results['pvalues'], pd.DataFrame):
            cate_output = format_output(cate_results['pvalues'], start, end,
                                        cate_results['n_variants'], n_voxels, cate)
            cate_output.to_csv(f'{args.out}_chr{chr}_start{start}_end{end}_{cate}.txt',
                               sep='\t', header=True, na_rep='NA', index=None, float_format='%.5e')