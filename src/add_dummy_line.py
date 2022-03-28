from fuc import pyvcf
import os
import sys
import subprocess
import pandas as pd

# to_add_vcf='/home/nguyen/deep_hla/test/GSAv3_24.GRCh38.rename.HLAregion.vcf.gz'
# query_vcf='/home/nguyen/deep_hla/test/consensus23.phased.HLA.vcf.gz'
# prefix='/home/nguyen/deep_hla/test/intersection'

PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
if  PARENT_PATH not in sys.path:
    sys.path.append(PARENT_PATH)

def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--to_add_vcf",type=str, 
                        default='/home/nguyen/deep_hla/test/GSAv3_24.GRCh38.rename.HLAregion.vcf.gz',
                        help="Source file for adding dummy line")
    parser.add_option("--query_vcf",type=str,
                        default='/home/nguyen/deep_hla/test/consensus23.phased.HLA.vcf.gz',
                        help="Original VCF")
    parser.add_option("--prefix",type=str,
                        default='/home/nguyen/deep_hla/test/intersection',
                        help='Prefix for output')
    (options, args) = parser.parse_args()
    return options

def bcftools_isec(to_add_vcf, query_vcf, prefix):

    bash_file = os.path.join(PARENT_PATH,'standardize_vcf.sh')
    cmd = ['bash', bash_file, to_add_vcf, query_vcf, prefix]

    sp = subprocess.Popen(cmd, shell=False)
    sp.wait()
    return sp.returncode

def create_dummy_df(private_df, query_df):
    # create dummy df
    dummy_df = pd.DataFrame(columns=query_df.df.columns,
                            index=private_df.df.index)

    dummy_df.iloc[:,:7] = private_df.df.iloc[:,:7].values
    dummy_df['INFO'] = 'AC=0;AF=0'
    dummy_df['FORMAT'] = 'GT'
    dummy_df.iloc[:, 9:] = '0|0'
    return dummy_df

def merge_dummy_line(to_add_vcf, query_vcf, prefix):
    returncode_isec = bcftools_isec(to_add_vcf, query_vcf, prefix)

    if returncode_isec == 0:
        print('Intersection finished')
        private_df = pyvcf.VcfFrame.from_file(prefix + '/0000.vcf')
        query_df = pyvcf.VcfFrame.from_file(query_vcf)
        
        dummy_df = create_dummy_df(private_df, query_df)
        merged_df = pd.concat([dummy_df, query_df.df], axis=0)
        merged_vcf = pyvcf.VcfFrame.from_dict(query_df.meta, merged_df)
        merged_sorted_vcf = merged_vcf.sort()
        merged_sorted_vcf.to_file(prefix + '/final.vcf')
    else:
        print('Something wrong in intersection') 


def main():
    options = get_options()
    merge_dummy_line(options.to_add_vcf, options.query_vcf, options.prefix)

if __name__ == "__main__":
    main()