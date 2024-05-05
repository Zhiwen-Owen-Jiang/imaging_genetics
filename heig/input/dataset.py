import sys
import os
import pickle
import logging
import pandas as pd
import numpy as np
from heig import utils


class Dataset():
    def __init__(self, dir):
        """
        Read a dataset and do preprocessing

        Parameters:
        ------------
        dir: diretory to the dataset

        """
        self.logger = logging.getLogger(__name__)
        if dir.endswith('dat'):
            self.logger.info(
                'WARNING: reading a .dat file is deprecated. Will be removed in the next version.')
            self.data = pickle.load(open(dir, 'rb'))
            if not 'FID' in self.data.columns or not 'IID' in self.data.columns:
                raise ValueError('FID and IID do not exist')
        else:
            openfunc, compression = utils.check_compression(dir)
            self._check_header(openfunc, compression, dir)
            self.data = pd.read_csv(dir, sep='\s+', compression=compression,
                                    na_values=[-9, 'NONE', '.'], dtype={'FID': str, 'IID': str})
        n_sub = len(self.data)
        self.data.drop_duplicates(
            subset=['FID', 'IID'], inplace=True, keep=False)
        self.logger.info(
            f'Removed {n_sub - len(self.data)} duplicated subjects.')
        self._remove_na_inf()

        self.data = self.data.set_index(['FID', 'IID'])
        self.data = self.data.sort_index()

    def _check_header(self, openfunc, compression, dir):
        """
        The dataset must have a header: FID, IID, ...

        Parameters:
        ------------
        openfunc: function to read the first line
        compression: how the data is compressed
        dir: diretory to the dataset

        """
        with openfunc(dir, 'r') as file:
            header = file.readline().split()
        if len(header) == 1:
            raise ValueError(
                'only one column detected, check your input file and delimiter')
        if compression is not None:
            header = [str(header[i], 'UTF-8') for i in range(len(header))]
        if header[0] != 'FID' or header[1] != 'IID':
            raise ValueError('the first two column names must be FID and IID')
        if len(header) != len(set(header)):
            raise ValueError('duplicated column names are not allowed')

    def _remove_na_inf(self):
        """
        Removing rows with any missing/infinite values

        """
        bad_idxs = (self.data.isin([np.inf, -np.inf, np.nan])).any(axis=1)
        self.data = self.data.loc[~bad_idxs]
        self.logger.info(
            f"Removed {sum(bad_idxs)} row(s) with missing or infinite values.")

    def keep(self, idx):
        """
        Extracting rows using indices (not boolean)
        the resulting dataset should have the same order as the indices

        Parameters:
        ------------
        idx: indices of the data

        """
        self.data = self.data.loc[idx]
        if len(self.data) == 0:
            raise ValueError('no data left')


class Covar(Dataset):
    def __init__(self, dir, cat_covar_list=None):
        """
        Parameters:
        ------------
        dir: diretory to the dataset
        cat_covar_list: a string of categorical covariates separated by comma

        """
        super().__init__(dir)
        self.cat_covar_list = cat_covar_list

    def cat_covar_intercept(self):
        """
        Converting categorical covariates to dummy variables,
        and adding the intercept. This step must be done after
        merging datasets, otherwise some dummy variables might
        cause singularity.

        """
        if self.cat_covar_list is not None:
            catlist = self.cat_covar_list.split(',')
            self._check_validcatlist(catlist)
            self.logger.info(
                f"{len(catlist)} categorical variables provided by --cat-covar-list.")
            self.data = self._dummy_covar(catlist)
        if not self.data.shape[1] == self.data.select_dtypes(include=np.number).shape[1]:
            raise ValueError('did you forget some categorical variables?')
        self._add_intercept()
        if self._check_singularity():
            raise ValueError('the covarite matrix is singular')

    def _check_validcatlist(self, catlist):
        """
        Checking if all categorical covariates exist

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        for cat in catlist:
            if cat not in self.data.columns:
                raise ValueError(f"{cat} cannot be found in the covariates")

    def _dummy_covar(self, catlist):
        """
        Converting categorical covariates to dummy variables

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        covar_df = self.data[catlist]
        qcovar_df = self.data[self.data.columns.difference(catlist)]
        if not qcovar_df.shape[1] == qcovar_df.select_dtypes(include=np.number).shape[1]:
            raise ValueError('did you forget some categorical variables?')
        covar_df = pd.get_dummies(covar_df, drop_first=True).astype(int)
        data = pd.concat([covar_df, qcovar_df], axis=1)

        return data

    def _add_intercept(self):
        """
        Adding the intercept

        """
        n = self.data.shape[0]
        const = pd.Series(np.ones(n), index=self.data.index, dtype=int)
        self.data = pd.concat([const, self.data], axis=1)

    def _check_singularity(self):
        """
        Checking if a matrix is singular
        True means singular

        """
        if len(self.data.shape) == 1:
            return self.data == 0
        else:
            return np.linalg.cond(self.data) >= 1/sys.float_info.epsilon


def get_common_idxs(*idx_list):
    """
    Get common indices among a list of double indices for subjects.
    Each element in the list must be a pd.MultiIndex instance.

    """
    common_idxs = None
    for idx in idx_list:
        if idx is not None:
            if not isinstance(idx, pd.MultiIndex):
                raise TypeError('index must be a pd.MultiIndex instance')
            if common_idxs is None:
                common_idxs = idx.copy()
            else:
                common_idxs = common_idxs.intersection(idx)
    if common_idxs is None:
        raise ValueError('no valid index provided')
    if len(common_idxs) == 0:
        raise ValueError('no common index exists')

    return common_idxs


def read_geno_part(dir):
    _, compression = utils.check_compression(dir)
    genome_part = pd.read_csv(dir, header=None, sep='\s+', usecols=[0, 1, 2],
                              compression=compression)
    if not (genome_part[0] % 1 == 0).all():
        raise TypeError(('the 1st column in the genome partition file must be integers. '
                         'Check if a header is included and/or if chromosome X/Y is included'))
    if not ((genome_part[1] % 1 == 0) & (genome_part[2] % 1 == 0)).all():
        raise TypeError(
            ('the 2nd and 3rd columns in the genome partition file must be integers'))

    return genome_part


def read_keep(keep_files):
    """
    Keep common subject IDs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    Error out if no common IDs exist

    Parameters:
    ------------
    keep_files: a list of tab/white-delimited files

    Returns:
    ---------
    keep_idvs_: pd.MultiIndex of common subjects

    """
    for i, keep_file in enumerate(keep_files):
        if os.path.getsize(keep_file) == 0:
            continue
        _, compression = utils.check_compression(keep_file)
        keep_idvs = pd.read_csv(keep_file,
                                sep='\s+',
                                header=None,
                                usecols=[0, 1],
                                dtype={0: str, 1: str},
                                compression=compression)
        keep_idvs = pd.MultiIndex.from_arrays([keep_idvs[0],
                                               keep_idvs[1]],
                                              names=['FID', 'IID'])
        if i == 0:
            keep_idvs_ = keep_idvs.copy()
        else:
            keep_idvs_ = keep_idvs_.intersection(keep_idvs)

    if len(keep_idvs_) == 0:
        raise ValueError('no subjects are common in --keep')

    return keep_idvs_


def read_extract(extract_files):
    """
    Keep common SNPs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    Error out if no common SNPs exist

    Parameters:
    ------------
    extract_files: a list of tab/white-delimited files

    Returns:
    ---------
    keep_snp_: pd.DataFrame of common SNPs

    """
    for i, extract_file in enumerate(extract_files):
        if os.path.getsize(extract_file) == 0:
            continue
        _, compression = utils.check_compression(extract_file)
        keep_snps = pd.read_csv(extract_file,
                                sep='\s+',
                                header=None,
                                usecols=[0],
                                names=['SNP'],
                                compression=compression)
        if i == 0:
            keep_snps_ = keep_snps.copy()
        else:
            keep_snps_ = keep_snps_.merge(keep_snps)

    if len(keep_snps_) == 0:
        raise ValueError('no SNPs are common in --extract')

    return keep_snps_
