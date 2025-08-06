"""Setup configuration for this package."""

from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="drupal-data-fetcher",
    version="1.0.0",
    install_requires=requirements,
    packages=find_packages(),
    author="Mattieu Scarset",
    author_email="m@matthieuscarset.com",
    license="MIT"
)
