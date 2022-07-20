# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "swagger_server"
VERSION = "1.0.0"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = [
    "connexion",
    "swagger-ui-bundle>=0.0.2"
]

setup(
    name=NAME,
    version=VERSION,
    description="API de pseudonymisation en langue française",
    author_email="",
    url="https://datascience.etalab.studio/pseudo/",
    keywords=["Swagger", "API de pseudonymisation en langue française"],
    install_requires=REQUIRES,
    packages=find_packages(),
    package_data={'': ['swagger/swagger.yaml']},
    include_package_data=True,
    entry_points={
        'console_scripts': ['swagger_server=swagger_server.__main__:main']},
    long_description="""\
    Cette API repose sur la reconnaissance d&#x27;entités nommées. Elle prend en entrée des textes en Français, et renvoie des textes modifiés où les entités détectées ont été mises entre balises et pseudonymisées (c&#x27;est-à-dire remplacées par d&#x27;autres valeurs).
    """
)
