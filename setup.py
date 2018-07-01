# coding=utf-8

from setuptools import setup
from setuptools import find_packages

setup(name='deepyeast',
      version='0.1',
      description='Deep Learning for Microscopy Data',
      url='http://github.com/tanelp/deepyeast',
      author='Tanel Pärnamaa',
      author_email='tanel.parnamaa@gmail.com',
      license='MIT',
      install_requires=['keras>=2.0.8'],
      packages=find_packages())
