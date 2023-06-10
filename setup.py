import os
from setuptools import setup, find_packages


def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)


# Project description
descr = 'Python library implementing linear operators with MPI'

# Setup
setup(
    name='pylops_mpi',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=['numpy >= 1.15.0', 'scipy >= 1.4.0', 'pylops >= 2.0',
                      'mpi4py', 'matplotlib'],
    packages=find_packages(exclude=['tests']),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('pylops_mpi/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='tests',
    tests_require=['pytest'],
    zip_safe=True)
