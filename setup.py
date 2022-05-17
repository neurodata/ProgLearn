import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "proglearn", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", mode="r", encoding="utf8") as f:
    REQUIREMENTS = f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")
        version = "v{}".format(VERSION)

        if tag != version:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, version
            )
            sys.exit(info)


setup(
    name="proglearn",
    version=VERSION,
    author="Will LeVine, Jayanta Dey, Hayden Helm",
    author_email="levinewill@icloud.com",
    maintainer="Jayanta Dey, Haoyin Xu",
    maintainer_email="jdey4@jhmi.edu",
    description="A package to implement and extend the methods desribed in 'Omnidirectional Transfer for Quasilinear Lifelong Learning'",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/ProgLearn/",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True,
    cmdclass={"verify": VerifyVersionCommand,},
)
