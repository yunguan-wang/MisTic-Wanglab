"""The Command-Line Interface (CLI) of MisC

The CLI of MisC can be accessed via ``python -m MisC``.

:Example:

    Get help:
    
    .. code-block:: bash

        python -m MisC -h
    
    Check version and authors:
    
    .. code-block:: bash
    
        python -m MisC --version 
        python -m MisC --author

"""


import os
import sys
import torch
import argparse

from MisC import __version__, __author__
from MisC.misc_class import misc

parser = argparse.ArgumentParser(description="Misc")

parser.add_argument("--version", action="version",
                    version=__version__, help="Display the version of the software")
parser.add_argument("--author", action="version", version=__author__,
                    help="Check the author list of the algorithm")


# Maybe consider
# config.yaml


def main(cmdargs: argparse.Namespace):
    """The main method for MisC

    Parameters:
    ----------
    cmdargs: argparse.Namespace
        The command line argments and flags 
    """
    

    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)