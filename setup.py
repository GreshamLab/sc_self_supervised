import os
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas",
    "anndata>=0.8",
    "scanpy",
    "scipy",
    "scikit-learn",
    "tqdm",
    "sparse_dot_mkl",
    "numba"
]

tests_require = [
    "coverage",
    "pytest"
]

version = "0.3.0"

# Description from README.md
long_description = "\n\n".join(
    [open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "README.md"
        ),
        "r"
    ).read()]
)

setup(
    name="scself",
    version=version,
    description="Self Supervised Tools for Single Cell Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GreshamLab/sc_self_supervised",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=[
        "scself",
        "scself.*"
    ]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="pytest",
)
