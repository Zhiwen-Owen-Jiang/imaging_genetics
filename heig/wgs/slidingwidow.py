import h5py
import numpy as np
import pandas as pd
import hail as hl
from heig.wgs.staar import VariantSetTest
from heig.wgs.coding import (
    extract_variant_type, extract_gene, Annotation_name_catalog, Annotation_name
)



class SlidingWindow:
    def __init__(self, snps, variant_type):
        self.snps = snps
        self.variant_type = variant_type

    def get_annotation(self):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        """
        if self.variant_type != 'snv':
            anno_phred = self.snps.fa.annotate(null_weight=1)
        else:
            anno_cols = [Annotation_name_catalog[anno_name]
                        for anno_name in Annotation_name]

            # anno_phred = self.snps.fa[anno_cols].to_pandas()
            # anno_phred['cadd_phred'] = anno_phred['cadd_phred'].fillna(0)
            # anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
            # anno_phred['apc_local_nucleotide_diversity2'] = anno_local_div
            
            anno_phred = self.snps.fa.select(*anno_cols)
            anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
            anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
            anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div)    
        return anno_phred
    


