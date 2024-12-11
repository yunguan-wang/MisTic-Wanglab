from setuptools import setup 
import re 


VERSIONFILE="MisC/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
# Version 
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
# Authors
AURE = r"^__author__ = ['\"]([^'\"]*)['\"]"
if mo:
    AUTHORS = mo.group(1)
else:
    raise RuntimeError("Unable to find author string in %s." % (VERSIONFILE,))
# Description
DESRE = r"^__description__ = ['\"]([^'\"]*)['\"]"
if mo:
    DESCRIPTION = mo.group(1)
else:
    raise RuntimeError("Unable to find description string in %s." % (VERSIONFILE,))
# Title 
TITRE = r"^__title__ = ['\"]([^'\"]*)['\"]"
if mo:
    TITLE = mo.group(1)
else:
    raise RuntimeError("Unable to find title string in %s." % (VERSIONFILE,))

# Title 
AUEMRE = r"^__authoremail__ = ['\"]([^'\"]*)['\"]"
if mo:
    AUTHOREMAIL = mo.group(1)
else:
    raise RuntimeError("Unable to find author email string in %s." % (VERSIONFILE,))


setup(
    name=TITLE,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHORS,
    author_email=AUTHOREMAIL,
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=["MisC"],
    python_requires=">=3.9,<3.11",
    install_requires=[
        'adjustText==1.2.0',
        'h5py==3.11.0',
        'matplotlib==3.9.1',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'pydeseq2==0.4.12',
        'scanpy==1.10.2',
        'scikit-learn==1.5.1',
        'scipy==1.13.1',
        'seaborn==0.13.2',
        'shapely==2.0.4',
        'squidpy==1.5.0',
        'tables==3.9.2',
        'torch==1.13.0',
        'tqdm==4.66.4',
    ],
    test_requires=[
        'pytest==7.1.2',
        'coverage==6.3.2',
        'pytest-cov==3.0.0',
        'pytest-mock==3.10.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ]
)