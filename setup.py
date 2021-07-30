from setuptools import setup, find_packages

setup(
    name='BlooPy',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'bitarray',
        'pyswarms'],
)
