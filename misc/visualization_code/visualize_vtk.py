import argparse
import pandas as pd


def main(args):
    print(f'Read result file from {args.res}')
    res_file = pd.read_csv(args.res, sep='\t')

    print(f'Read vtk file from {args.vtk}')
    with open(args.vtk, 'r') as input:
        tempB = input.readlines()
    
    args.col_name = args.col_name.split(',')
    for col in args.col_name:
        if hasattr(res_file, col):
            res = res_file[col]
        else:
            print(f'WARNING: {args.col} cannot be found. Skip.')

        tempB.append(f"SCALARS {col} float\n")
        tempB.append(f"LOOKUP_TABLE {col}\n")
        
        for num in res:
            tempB.append(f"{str(num)}\n")
        
    with open(f"{args.out}.vtk", 'w') as output:
        output.writelines(tempB)
    print(f"Save the vtk file to {args.out}.vtk")


parser = argparse.ArgumentParser()
parser.add_argument('--res', help='a white space-delimited result file')
parser.add_argument('--vtk', help='a vtk file')
parser.add_argument('--out', help='output (prefix)')
parser.add_argument('--col-name', help=('which column name(s) in the result file '
                                        'you want to visualize. '
                                        'Multiple names should be separated by comma, '
                                        'e.g. col1,col2,col3'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
