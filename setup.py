# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='balancemm',
    version='0.1', 
    description='A Benchmark for balanced multimodal learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HPLQAQ/BalanceMM/',
    author='Siwei Li',
    author_email='hplv@foxmail.com', 
    classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
    ],
    keywords='multimodal, benchmark, balanced learning, classification, deep learning',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'docstring-parser',
        'lightning',
        'torch',
        'torchvision',
        'torchaudio',
        'xformers',
    ],
)