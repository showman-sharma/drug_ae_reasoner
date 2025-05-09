import os
from setuptools import setup, find_packages

setup(
    name="drug_ae_reasoner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "numpy",
        "faiss-cpu",
        "networkx",
    ],
    entry_points={
        'console_scripts': [
            'drug_ae_reasoner=drug_ae_reasoner.main:main',
        ],
    },
    author="Anirudh Sharma",
    description="A package to reason drug-AE connections using RxNorm, CADEC, and OAE.",
    long_description = open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
