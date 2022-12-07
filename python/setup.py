# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='partorch',
      version='0.1',
      description='Example of how to make models parallel in pytorch',
      url='https://github.com/enccs/partorch',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['partorch'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)
