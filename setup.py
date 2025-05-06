from setuptools import setup 
import re 
import os 


VERSIONFILE="MisTIC/__version__.py"
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
mo = re.search(AURE, verstrline, re.M)
if mo:
    AUTHORS = mo.group(1)
else:
    raise RuntimeError("Unable to find author string in %s." % (VERSIONFILE,))
# Description
DESRE = r"^__description__ = ['\"]([^'\"]*)['\"]"
mo = re.search(DESRE, verstrline, re.M)
if mo:
    DESCRIPTION = mo.group(1)
else:
    raise RuntimeError("Unable to find description string in %s." % (VERSIONFILE,))
# Title 
TITRE = r"^__title__ = ['\"]([^'\"]*)['\"]"
mo = re.search(TITRE, verstrline, re.M)
if mo:
    TITLE = mo.group(1)
else:
    raise RuntimeError("Unable to find title string in %s." % (VERSIONFILE,))

# Title 
AUEMRE = r"^__authoremail__ = ['\"]([^'\"]*)['\"]"
mo = re.search(AUEMRE, verstrline, re.M)
if mo:
    AUTHOREMAIL = mo.group(1)
else:
    raise RuntimeError("Unable to find author email string in %s." % (VERSIONFILE,))


lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] 
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()


setup(
    name=TITLE,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHORS,
    author_email=AUTHOREMAIL,
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=["MisTIC"],
    python_requires=">=3.9,<3.11",
    install_requires=install_requires,
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