#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os.path
import sys

import setuptools
from setuptools.command.develop import develop as DefaultDevelopCommand
from setuptools.command.install import install as DefaultInstallCommand

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "habitat"))
from version import VERSION  # isort:skip noqa


with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_text = f.read()

DISTNAME = "habitat"
DESCRIPTION = "habitat: a suite for embodied agent tasks and benchmarks"
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = license_text
DEFAULT_EXCLUSION = ["test", "examples"]

REQUIREMENTS = set()
for file_name in glob.glob("habitat_baselines/**/requirements.txt", recursive=True):
    with open(file_name) as f:
        reqs = f.read()
        REQUIREMENTS.update(reqs.strip().split("\n"))
REQUIREMENTS = list(REQUIREMENTS)

if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(exclude=DEFAULT_EXCLUSION),
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["pytest-cov", "pytest-mock", "pytest"],
        include_package_data=True,
        cmdclass={"install": DefaultInstallCommand, "develop": DefaultDevelopCommand},
    )
