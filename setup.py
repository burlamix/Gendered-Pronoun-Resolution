#!/usr/bin/env python3
"""hltproject setup.py.

This file details modalities for packaging the hltproject application.
"""

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='hltproject',
    description='model to solve coreference tasks on Gendered Ambiguous Pronoun (GAP) datasets',
    author='Lucio Messina, Simone Spagnoli, Gaspare Ferraro',
    author_email='lucio.messina@autistici.org',
    long_description=long_description,
    version='0.1.0',
    url='https://github.com/burlamix/Gendered-Pronoun-Resolution',
    download_url='https://github.com/burlamix/Gendered-Pronoun-Resolution',
    license='GPL 3.0',
    keywords=[],
    platforms=['any'],
    packages=['hltproject', 'hltproject.dataset_utils', 'hltproject.baseline', 'hltproject.logging_config',
              'hltproject.score'],
    package_data={'hltproject': ['logging_config/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'hltproject = hltproject.main:main'
        ],
    },
    install_requires=['tqdm==4.31.1', 'torchvision==0.4.0', 'gender-guesser', 'pyyaml==3.13'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GPL 3.0 License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
