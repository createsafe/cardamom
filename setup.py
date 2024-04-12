from setuptools import setup, find_packages

setup(
    name='cardamom',
    version='0.1',
    author='Max Ardito, Alba Saco, Julian Vanasse',
    description='Not your average audio DSP library',
    url='https://github.com/createsafe/cardamom',
    packages=find_packages(),
    install_requires=['torch>=1.0'],  # Add any dependencies here
)
