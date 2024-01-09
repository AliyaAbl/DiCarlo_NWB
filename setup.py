from setuptools import setup, find_packages
import os

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as fb:
        requirements = fb.readlines()
else:
    requirements = []

setup(
    name="ndashboard",
    version="0.01.dev",
    packages=find_packages(),
    install_requires=requirements, # todo
    author="Michael Lee",
    author_email="mil@mit.edu",
    description="Tools for assessing quality of neural data",
    keywords="",
)
