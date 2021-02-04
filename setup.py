import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="nanograd",
    version="1.0.3",
    description="A lightweight deep learning framework",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PABannier/nanograd",
    author="PAB",
    author_email="pierre-antoine.bannier@polytechnique.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=('tests', 'examples', 'utils')),
    include_package_data=True,
    install_requires=[
        "numpy", 
        "matplotlib", 
        "graphviz", 
        "pyopencl", 
        "mako"
    ],
)