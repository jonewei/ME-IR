from setuptools import setup, find_packages

setup(
    name="math-expression-retrieval",
    version="0.1.0",
    description="Hierarchical and graph-aware mathematical expression retrieval",
    author="JONEWEI",
    author_email="your.email@university.edu",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "torch",
        "transformers",
        "tqdm",
        "networkx",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Information Retrieval",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)


# pip install -e .
