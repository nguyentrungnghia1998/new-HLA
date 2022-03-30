from importlib.metadata import entry_points
from setuptools import setup
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup (
    name = 'HLA tool',
    version = '0.1',
    description = 'A tool for predicting HLA alleles',
    long_description = long_description,
    long_description_content_type="text/markdown", 
    author = 'Vu Quoc Hien',
    author_email = 'v.hienvq1@vinbigdata.org',
    url = 'https://bitbucket.org/Hienvq2304/deep_hla',
    packages=["src", "pipelines", "models", "objects"],
    keywords='hla, deep learning, allele prediction',
    install_requires=[
        'cyvcf2==0.30.14',
        'matplotlib==3.5.1',
        'numpy==1.21.2',
        'pandas==1.3.3',
        'torch==1.9.1',
        'tqdm==4.62.3',
        'torchsummary==1.5.1',
        'scikit-learn==1.0.2',
        'seaborn==0.11.2',
        'fuc==0.31.0'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'hla-train=pipelines.hla_train:main',
            'hla-test=pipelines.hla_test:main',
            'hla-tool=pipelines.hla_tool:main'
        ]
    },
    
)