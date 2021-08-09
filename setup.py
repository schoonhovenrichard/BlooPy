from setuptools import setup, find_packages

setup(
    name = 'bloopy',
    packages=find_packages(),
    version = '0.4',  # Ideally should be same as your GitHub release tag varsion
    description = 'BlooPy: Black-box optimization Python for bitstring, categorical, and numerical discrete problems with local, and population-based algorithms.',
    author = 'Richard Schoonhoven',
    author_email = 'r.a.schoonhoven@hotmail.com',
    url = 'https://github.com/schoonhovenrichard/BlooPy',
    download_url = 'https://github.com/schoonhovenrichard/BlooPy/archive/refs/tags/0.3.tar.gz',
    keywords = ["auto-tuning","optimization","gradient-free","black-box","computing","algorithms","discrete","minimization","maximization","evolutionary","fitness"],
    classifiers = [],
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'bitarray',
        'pyswarms',
        'networkx',
        'pytest',
        'hypothesis'],
)
