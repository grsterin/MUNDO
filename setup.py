from setuptools import setup, find_packages

setup(
    name='gmundo',
    version='1',
    description='Multi-MUNDO',
    author='Grigorii Sterin and Kapil Devkota',
    author_email='grigorii.sterin@tufts.edu',
    url='https://github.com/grsterin/MUNDO.git',
   # packages=find_packages(exclude=('tests', 'docs', 'results', 'data')),
    packages = ['gmundo', 'gmundo.prediction'],
    package_dir={'gmundo':'gmundo'}
    ##  package_data={'glide':['data/*.dat']}
)
