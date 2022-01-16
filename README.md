# README #
### Sampling 100
```
bcftools query -l consensus23.phased.HLA.vcf.gz | head -n 100 > test.list

bcftools view -S test.list consensus23.phased.HLA.vcf.gz --force-samples  -Oz -o test_subsample.vcf.gz
```
