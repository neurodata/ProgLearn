from setuptools import setup, find_packages

requirements = [
    "keras",
    "tensorflow ",
    "scikit-learn",
    "numpy",
    "joblib",
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="proglearn",
    version="0.0.1",
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
        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True
)
