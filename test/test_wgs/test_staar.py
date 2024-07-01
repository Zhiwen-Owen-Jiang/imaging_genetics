import unittest
import os, sys
import numpy as np
import hail as hl
from hail.linalg import BlockMatrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from heig.wgs.staar import VariantSetTest
from heig.wgs.utils import preprocess



if __name__ == '__main__':
    # snps_mt = hl.import_plink(bed='test/test_input/plink/plink.bed',
    #                           bim='test/test_input/plink/plink.bim',
    #                           fam='test/test_input/plink/plink.fam',
    #                           reference_genome='GRCh37')
    # snps_mt = preprocess(snps_mt, start=51065600, end=51094926, maf_thresh=0.5)
    # main = '/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis'
    # snps_mt = hl.import_plink(bed=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.bed',
    #                    bim=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.bim',
    #                    fam=f'{main}/bfiles/bfiles_6m/ukb_imp_chr21_v3_maf_hwe_INFO_QC_white_phase123_nomulti.fam',
    #                    reference_genome='GRCh37')
    # snps_mt = hl.read_matrix_table('/work/users/o/w/owenjf/hail/ukb_call_chr21.mt')
    # snps_mt = preprocess(snps_mt, maf_thresh=0.5, start=14677301, end=20298186)
    snps_mt = hl.import_plink(bed='/work/users/o/w/owenjf/hail/ukb_call_chr21_50k.bed',
                              bim='/work/users/o/w/owenjf/hail/ukb_call_chr21_50k.bim',
                              fam='/work/users/o/w/owenjf/hail/ukb_call_chr21_50k.fam',
                              reference_genome='GRCh37')
    snps_mt = preprocess(snps_mt, maf_thresh=0.02)
    bases = np.load('test/test_secondary/bases.npy')
    covar = np.ones(50000).reshape(-1, 1)
    ldr = np.random.randn(200000).reshape(50000, 4)
    resid_ldr = ldr - np.mean(ldr, axis=0)
    inner_ldr = np.dot(resid_ldr.T, resid_ldr)
    var = np.random.randn(5) ** 2

    vtest = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    vtest.input_vset(snps_mt)
    vtest.do_inference()
   


