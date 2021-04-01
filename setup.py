from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os
import sys
import platform

if platform.python_implementation() == "PyPy":
    SCIPY_MIN_VERSION = "1.1.0"
    NUMPY_MIN_VERSION = "1.14.0"
else:
    SCIPY_MIN_VERSION = "0.17.0"
    NUMPY_MIN_VERSION = "1.11.0"

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    "develop",
    "release",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wininst",
    "install_egg_info",
    "build_sphinx",
    "egg_info",
    "easy_install",
    "upload",
    "bdist_wheel",
    "--single-version-externally-managed",
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            "alldeps": (
                "numpy >= {}".format(NUMPY_MIN_VERSION),
                "scipy >= {}".format(SCIPY_MIN_VERSION),
            ),
        },
    )
else:
    extra_setuptools_args = dict()

# Find mgc version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "proglearn", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", mode="r", encoding="utf8") as f:
    REQUIREMENTS = f.read()

# Cythonize splitter
ext_modules = [
    Extension(
        "split",
        ["proglearn/split.pyx"],
        extra_compile_args=[
            "-Xpreprocessor",
            "-fopenmp",
        ],
        extra_link_args=[
            "-Xpreprocessor", 
            "-fopenmp"
        ],
        language="c++",
    )
]


setup(
    name="proglearn",
    version=VERSION,
    author="Will LeVine, Jayanta Dey, Hayden Helm",
    author_email="levinewill@icloud.com",
    maintainer="Will LeVine, Jayanta Dey",
    maintainer_email="levinewill@icloud.com",
    description="A package to implement and extend the methods desribed in 'A General Approach to Progressive Learning'",
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
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
)
