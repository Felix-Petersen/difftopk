import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='difftopk',
    version='0.2.0',
    author='Felix Petersen',
    author_email='ads0600@felix-petersen.de',
    description='Differentiable Sorting, Ranking, and Top-k.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Felix-Petersen/difftopk',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    package_dir={'difftopk': 'difftopk'},
    packages=['difftopk'],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.6.0',
        'numpy',
        'diffsort',
    ],
)
