# bash src/standardize_vcf.sh 
to_add_vcf=$1
query_vcf=$2
prefix=$3

bcftools isec $to_add_vcf $query_vcf -p $prefix 