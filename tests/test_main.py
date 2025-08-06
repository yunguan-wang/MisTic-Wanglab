from MisTIC import __main__
import MisTIC
import pytest
import numpy as np

import sys
import os
import shutil
from io import StringIO

from typing import Dict, Union


def test_parser_version():
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    try:
        __main__.parser.parse_args(["--version"])
    except SystemExit:
        output = string_stdout.getvalue()
        expected = MisTIC.__version__.__version__ + "\n"
        assert output == expected
        sys.stdout = screen_stdout
    else:
        sys.stdout = screen_stdout
        assert False
        

@pytest.mark.parametrize("cmdarg", ["--help", "-h"])
def test_parser_help(cmdarg):
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    try:
        __main__.parser.parse_args([cmdarg])
    except SystemExit:
        output = string_stdout.getvalue()
        assert "usage: " in output
        sys.stdout = screen_stdout
    else:
        sys.stdout = screen_stdout
        assert False



