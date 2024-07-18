import pandas as pd
import numpy as np
import bitarray as ba
from heig import utils


"""
Credits to Brendan Bulik-Sullivan and the LDSC developers for the following functions
https://github.com/bulik/ldsc

"""


def __ID_List_Factory__(colnames, keepcol, id_dtypes, fname_end, header=None, usecols=None):

    class IDContainer:
        def __init__(self, fname):
            self.__usecols__ = usecols
            self.__colnames__ = colnames
            self.__keepcol__ = keepcol
            self.__fname_end__ = fname_end
            self.__header__ = header
            self.__id_dtypes = id_dtypes
            self.__read__(fname)
            self.n = len(self.df)

        def __read__(self, fname):
            end = self.__fname_end__
            if end and not fname.endswith(end):
                raise ValueError(f"{fname} must end in {end}")

            _, comp = utils.check_compression(fname)
            self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__,
                                  sep='\s+', compression=comp, dtype=self.__id_dtypes)

            if self.__colnames__:
                self.df.columns = self.__colnames__

            if self.__keepcol__ is not None:
                self.IDList = self.df.iloc[:, self.__keepcol__].astype('object')

    return IDContainer


PlinkBIMFile = __ID_List_Factory__(['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'], 1, {1: str},
                                   '.bim', usecols=[0, 1, 2, 3, 4, 5])
PlinkFAMFile = __ID_List_Factory__(['FID', 'IID', 'SEX'], [0, 1], {0: str, 1: str},
                                   '.fam', usecols=[0, 1, 4])
# FilterFile = __ID_List_Factory__(['ID'], 0, {0: str}, None, usecols=[0])


class __GenotypeArrayInMemory__:
    '''
    Parent class for various classes containing inferences for files with genotype
    matrices, e.g., plink .bed files, etc
    '''

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self.m = len(snp_list)
        self.n = n
        self.keep_snps = keep_snps
        self.keep_indivs = keep_indivs
        self.df = snp_list
        self.colnames = ['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2']
        self.mafMin = mafMin if mafMin is not None else 0
        self._currentSNP = 0
        (self.nru, self.geno) = self.__read__(fname, n)

        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype='int')
            if np.any(keep_indivs > self.n):
                raise ValueError('keep_indivs indices out of bounds')

            (self.geno, self.m, self.n) = self.__filter_indivs__(
                self.geno, keep_indivs, self.m)
            if self.n <= 0:
                raise ValueError('After filtering, no individuals remain')

        # filter SNPs
        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype='int')
            if np.any(keep_snps > self.m):  # if keep_snps is None, this returns False
                raise ValueError('keep_snps indices out of bounds')

        (self.geno, self.m, self.n, self.kept_snps, self.freq) = self.__filter_snps_maf__(
            self.geno, self.m, self.n, self.mafMin, keep_snps)

        if self.m <= 0:
            raise ValueError('After filtering, no SNPs remain')

        self.df = self.df.loc[self.kept_snps]
        self.maf = np.minimum(self.freq, np.ones(self.m)-self.freq)
        self.sqrtpq = np.sqrt(self.freq*(np.ones(self.m)-self.freq))
        self.df['MAF'] = self.maf
        self.colnames.append('MAF')

    def __read__(self):
        raise NotImplementedError

    def __filter_indivs__(self):
        raise NotImplementedError

    def __filter_snps_maf__(self):
        raise NotImplementedError


class PlinkBEDFile(__GenotypeArrayInMemory__):
    '''
    Interface for Plink .bed format
    '''

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self._bedcode = {
            2: ba.bitarray('11'),
            np.nan: ba.bitarray('10'),
            1: ba.bitarray('01'),
            0: ba.bitarray('00')
        }

        __GenotypeArrayInMemory__.__init__(self, fname, n, snp_list, keep_snps=keep_snps,
                                           keep_indivs=keep_indivs, mafMin=mafMin)

    def __read__(self, fname, n):
        if not fname.endswith('.bed'):
            raise ValueError('.bed filename must end in .bed')

        with open(fname, 'rb') as fh:
            magicNumber = ba.bitarray(endian='little')
            magicNumber.fromfile(fh, 2)
            bedMode = ba.bitarray(endian='little')
            bedMode.fromfile(fh, 1)
            e = (4 - n % 4) if n % 4 != 0 else 0
            nru = n + e
            self.nru = nru

            # check magic number
            if magicNumber != ba.bitarray('0011011011011000'):
                raise IOError('Magic number from PLINK .bed file not recognized')

            if bedMode != ba.bitarray('10000000'):
                raise IOError('Plink .bed file must be in default SNP-major mode')

            # check file length
            self.geno = ba.bitarray(endian='little')
            self.geno.fromfile(fh)
            self.__test_length__(self.geno, self.m, self.nru)
            return (self.nru, self.geno)

    def __test_length__(self, geno, m, nru):
        exp_len = 2*m*nru
        real_len = len(geno)
        if real_len != exp_len:
            raise IOError(f"Plink .bed file has {real_len} bits, expected {exp_len}")

    def __filter_indivs__(self, geno, keep_indivs, m):
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m*2*nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2*e::2*nru_new] = geno[2*i::2*nru]
            z[2*e+1::2*nru_new] = geno[2*i+1::2*nru]

        self.nru = nru_new
        return (z, m, n_new)

    def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
        '''
        Credit to Chris Chang and the Plink2 developers for this algorithm
        Modified from plink_filter.c
        https://github.com/chrchang/plink-ng/blob/master/plink_filter.c

        Genotypes are read forwards (since we are cheating and using endian="little")

        A := (genotype) & 1010...
        B := (genotype) & 0101...
        C := (A >> 1) & B

        Then

        a := A.count() = missing ct + hom major ct
        b := B.count() = het ct + hom major ct
        c := C.count() = hom major ct

        Which implies that

        missing ct = a - c
        # of indivs with nonmissing genotype = n - a + c
        major allele ct = b + c
        major allele frequency = (b+c)/(2*(n-a+c))
        het ct + missing ct = a + b - 2*c

        Why does bitarray not have >> ????

        '''
        nru = self.nru
        m_poly = 0
        y = ba.bitarray()
        if keep_snps is None:
            keep_snps = range(m)
        kept_snps = []
        freq = []
        for j in keep_snps:
            z = geno[2*nru*j:2*nru*(j+1)]
            A = z[0::2]
            a = A.count()
            B = z[1::2]
            b = B.count()
            c = (A & B).count()
            major_ct = b + c  # number of copies of the major allele
            n_nomiss = n - a + c  # number of individuals with nonmissing genotypes
            f = major_ct / (2*n_nomiss) if n_nomiss > 0 else 0
            het_miss_ct = a+b-2*c  # remove SNPs that are only either het or missing
            if np.minimum(f, 1-f) > mafMin and het_miss_ct < n:
                freq.append(f)
                y += z
                m_poly += 1
                kept_snps.append(j)

        return (y, m_poly, n, kept_snps, freq)

    def nextSNPs(self, num, nona=False):
        '''
        Unpacks the binary array of genotypes and returns an n x num matrix of 
        genotypes for next SNPs, where n := number of samples.
        nona: if fill na as 0

        '''
        nona_map = {
            2: ba.bitarray('11'),
            0: ba.bitarray('10'),  # na is mapped to 0
            1: ba.bitarray('01'),
            0: ba.bitarray('00')
        }
        if nona:
            mapping = nona_map
        else:
            mapping = self._bedcode

        if self._currentSNP + num > self.m:
            raise ValueError(f"{num} SNPs requested, {self.m - self._currentSNP} SNPs remain")

        slice = self.geno[2*self._currentSNP * self.nru: 2*(self._currentSNP+num)*self.nru]
        snps = np.array(slice.decode(mapping), dtype=float).reshape((num, self.nru)).T
        snps = snps[:self.n]
        self._currentSNP += num

        return snps

    def gen_SNPs(self):
        """
        Never use it

        """
        for c in range(self.m):
            slice = self.geno[2*c*self.nru: 2*(c+1)*self.nru]
            X = np.array(slice.decode(self._bedcode), dtype=float)
            X = X[0:self.n]

            yield c, X


def read_plink(dir, keep_snps=None, keep_indivs=None, maf=None):
    """
    Read plink triplets with a subset of SNPs/individuals

    Parameters:
    ------------
    dir: prefix of plink triplets
    keep_snps:  rsID of SNPs to extract
    keep_indivs: ID of individuals to keep
    maf: minimum MAF of SNPs to extract

    Returns:
    ---------
    geno_array.df: pd.DataFrame of bim file
    fam: pd.DataFrame of fam file
    snp_getter: a generator of SNPs

    """
    array_file, array_obj = f"{dir}.bed", PlinkBEDFile
    snp_file, snp_obj = f"{dir}.bim", PlinkBIMFile
    ind_file, ind_obj = f"{dir}.fam", PlinkFAMFile

    array_snps = snp_obj(snp_file)
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)

    if keep_snps is not None:
        keep_snps_idxs = array_snps.df.index[array_snps.df['SNP'].isin(keep_snps['SNP'])]
    else:
        keep_snps_idxs = None

    array_indivs.df = array_indivs.df.set_index(['FID', 'IID'])
    if keep_indivs is not None:
        keep_indivs_idxs = np.array(range(n))[array_indivs.df.index.isin(keep_indivs)]
        fam = array_indivs.df.loc[keep_indivs]
    else:
        keep_indivs_idxs = None
        fam = array_indivs.df

    geno_array = array_obj(array_file, n, array_snps.df, keep_snps=keep_snps_idxs,
                           keep_indivs=keep_indivs_idxs, mafMin=maf)
    snp_getter = geno_array.nextSNPs

    return geno_array.df, fam, snp_getter
