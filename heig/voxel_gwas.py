import os
import numpy as np
import pandas as pd
import pickle


MASTHEAD = "***********************************************************************************\n"
MASTHEAD += "* Voxelwise GWAS using HEIG\n"
MASTHEAD += "***********************************************************************************"


def recover_se(bases, inner_ldr, n, ldr_beta, ztz_inv):
    bases = np.atleast_2d(bases)
    part1 = np.sum(np.dot(bases, inner_ldr) * bases, axis=1).reshape(1, -1) # 100*1
    part2 = np.dot(ldr_beta, bases.T) ** 2 # d*100
    se = np.sqrt(part1 * ztz_inv / n - part2 / n)
    return se


def main(args, log):
    inner_ldr = np.load(args.inner_ldr)
    bases = np.load(args.bases)
    ldr_gwas = pickle.load(open(args.ldr_gwas, 'rb')) 

    ldr_gwas.beta_df = ldr_gwas.beta_df.iloc[:, :args.n_ldrs]
    ldr_gwas.se_df = ldr_gwas.se_df.iloc[:, :args.n_ldrs]
    bases = bases[:, :args.n_ldrs]
    inner_ldr = inner_ldr[:args.n_ldrs, :args.n_ldrs]

    ldr_beta = np.array(ldr_gwas.beta_df)
    ldr_se = np.array(ldr_gwas.se_df)
    ldr_n = np.array(ldr_gwas.snp_info['N']).reshape(-1, 1)

    if args.start and args.end:
        chr, start = [int(x) for x in args.start.split(':')]
        chr_, end =[int(x) for x in args.end.split(':')]
        if chr != chr_:
            raise ValueError('CHR should be the same for start and end')

        idx = (ldr_gwas.snp_info['POS'] > start) & (ldr_gwas.snp_info['POS'] < end) & (ldr_gwas.snp_info['CHR'] == chr)
        ldr_beta = ldr_beta[idx]
        ldr_se = ldr_se[idx]
        ldr_n = ldr_n[idx]
        outpath = f"{args.out}_voxel{args.voxel}_start{args.start}_end{args.end}.txt"
        snp_info = ldr_gwas.snp_info.loc[idx].copy()
    else:
        outpath = f"{args.out}_voxel{args.voxel}.txt"
        snp_info = ldr_gwas.snp_info
    
    ztz_inv = np.mean((ldr_n * ldr_se ** 2 + ldr_beta**2) / np.diag(inner_ldr), axis=1)
    ztz_inv = ztz_inv.reshape(-1, 1)

    voxel_beta = np.dot(ldr_beta, bases[args.voxel].T)
    voxel_se = np.squeeze(recover_se(bases[args.voxel], inner_ldr, ldr_n, ldr_beta, ztz_inv))
    voxel_z = voxel_beta / voxel_se
    
    snp_info['BETA_HEIG'] = voxel_beta
    snp_info['SE_HEIG'] = voxel_se
    snp_info['Z_HEIG'] = voxel_z
    snp_info.to_csv(outpath, sep='\t', index=None, na_rep='NA')
    log.info(f"Save the output to {outpath}")


parser = argparse.ArgumentParser()
parser.add_argument('--n-ldrs', type=int, help='number of LDRs to use')
parser.add_argument('--ldr-gwas', help='directory to LDR gwas files (prefix)')
parser.add_argument('--bases', help='directory to bases')
parser.add_argument('--inner-ldr', help='directory to inner product of LDR')
parser.add_argument('--voxel', type=int, help='which voxel, 0 based index')
parser.add_argument('--start', help='start position')
parser.add_argument('--end', help='end position')
parser.add_argument('--out', help='directory for output')



if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}_voxel{args.voxel}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "Parsed arguments\n"
        options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
        header += '\n'.join(options).replace('True','').replace('False','')
        header = header+'\n'
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"Analysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
