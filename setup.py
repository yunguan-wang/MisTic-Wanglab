from setuptools import setup 
import sys 
import os 
import shutil
import distutils.cmd


VERSION = "0.0.1"

class PypiCommand(distutils.cmd.Command):
    
    description = "Build and upload for PyPI."
    user_options = []
    
    def initialize_options(self):
        pass
    
    
    def finalize_options(self):
        pass
    
    
    def run(self):
        try:
            shutil.rmtree("dist/")
        except FileNotFoundError:
            pass
        
        wheel_file = "MisC-{}-py3-none-any.whl".format(VERSION)
        tar_file = "MisC-{}.tar.gz".format(VERSION)
        
        os.system("{} setup.py sdist bdist_wheel".format(sys.executable))
        os.system("twine upload dist/{} dist/{}".format(wheel_file, tar_file))


setup(
    name="MisC",
    version=VERSION,
    description="Reassign transcripts",
    author="???",
    author_email="???",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=["Misc"],
    python_requires=">=3.9,<3.11",
    install_requires=[
        'numpy==1.22.4',
        'pandas==1.5.2',
        'tqdm==4.64.1',
        'torch==1.13.1',
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
    ],
    cmdclass={
        "pypi": PypiCommand,
    }
)