# README #

# DEEP-HLA IMPUTATION 

## Sampling data

To sampling data from vcf file, use bcftools, create file list of index. Test for subsample data before original data

### Install bcftools 
```
sudo apt-get update -y

sudo apt-get install -y bcftools
```
### Example for sample 100 individuals
```
cd input/

bcftools query -l consensus23.phased.HLA.vcf.gz | head -n 1020 > consensus23.phased.HLA.sample.list

bcftools query -f '%POS\n' consensus23.phased.HLA.vcf.gz > consensus23.phased.HLA.position.list

bcftools view -S test.list consensus23.phased.HLA.vcf.gz --force-samples  -Oz -o consensus23.phased.HLA.sample.vcf.gz
```

## Install Python Lib and package

To install neccessary package for project, pytorch, cy2vcf, ... They are included in `setup.py` file. Before install package, you need activate a conda environment.

### Install Miniconda and activate environment
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

conda create -n hla-tool python=3.8
```
### Install package

```
cd <Directory  where setup.py is located>
pip install .
```

## How to run project in Terminal

To create 2 folder trained_model, model output include figures, results of loss function, accuracy cross validation and evaluate test dataset. Trained model used to predict for actual data if test result is admissible

### Parameters in command
```
--pretrain if you only want to preprocess input data

--trainning if you only want to train model

--dataset-path: Path to processed vcf.gz file from bcftools

--index-path: Path to file list index of people from bcftools

--label-path: Path to file label for variant on each HLA

--sample: Number of feature (hla allele length) use for training model from file label, maximum 101000

--hla-types: List of HLAs used for training model, separated by commas

--model-save-dir: Path to save trained model

--output-path: Path to base folder, save figures, results

-l: Select loss function by string:
'mse': Mean squared error
'cross_entropy': Cross entropy loss
'bce': Binary cross entropy loss
'bce_logits': Use softmax function before binary cross entropy loss
'l1': L1 loss
'smooth_l1': Smooth L1 loss
'soft_margin': Soft margin loss

-o: Optimizer algorithm: 'sgd', 'adam', 'adadelta', 'adagrad', 'rmsprop'

--use_cross_validation: Determine if use cross validation or not, store_true (use cross validation), store_false (not use cross validation)

-k: Number of fold if use cross validation

-e: Number of train epochs

--lr: Learning rate 

-b: batch size

-v: Determine if display results after each epochs or not (train loss, validation loss, validation accuracy), store_true (display), store_false (not display)
```
### Run project and evaluate results

``` 
python main.py [define list of parameters]
```

Results in path, defined in --output-path variable, include figures of cross validation, final result.
Trained model saved in path, defined in --model-path variable.