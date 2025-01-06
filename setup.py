from setuptools import setup, find_packages

setup(
    name='engXPBD',
    version='0.1.0',
    description='A Python implementation of Extended Position-Based Dynamics (XPBD) for deformable solids.',
    author='Saman Seifi',
    author_email='saman.seyfi@gmail.com',
    licence='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'meshio',
        'scipy',
        'h5py',
    ],
)