from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rock-cd",
    version="0.1",
    author="Andrea Failla, Fedetico Mazzoni",
    author_email="andrea.failla.ak@gmail.com",
    description="ROCK-CD: an implementation of ROCK for Community Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andreafailla/rock-cd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords="community-discovery rock",
    python_requires=">=3.0",
)
