import gzip, bz2
import pandas as pd
import logging
from functools import reduce


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode='w')
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f


def check_compression(dir):
    """
    Checking which compression should use

    Parameters:
    ------------
    dir: diretory to the dataset

    Returns:
    ---------
    openfunc: function to open the file
    compression: type of compression
    
    """
    if dir.endswith('gz'):
        compression = 'gzip'
        openfunc = gzip.open
    elif dir.endswith('bz2'):
        compression = 'bz2'
        openfunc = bz2.BZ2File
    else:
        openfunc = open
        compression = None

    return openfunc, compression



def read_keep(keep_files):
    for i, keep_file in enumerate(keep_files):
        keep_idvs = pd.read_csv(keep_file, delim_whitespace=True, header=None, usecols=[0, 1],
                               dtype={0: str, 1: str})
        keep_idvs = pd.MultiIndex.from_arrays([keep_idvs[0], keep_idvs[1]], names=['FID', 'IID'])
        if i == 0:
            keep_idvs_ = keep_idvs.copy()
        else:
            keep_idvs_ = keep_idvs_.intersection(keep_idvs)
    
    return keep_idvs_ 



def read_extract(extract_files):
    for i, extract_file in enumerate(extract_files):
        keep_snps = pd.read_csv(extract_file, delim_whitespace=True, 
                               header=None, usecols=[0], names=['SNP']) 
        if i == 0:
            keep_snps_ = keep_snps.copy()
        else:
            keep_snps_ = keep_snps_.merge(keep_snps)

    return keep_snps_
        