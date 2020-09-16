from setuptools import setup, find_packages

requirements = [
    "keras",
    "tensorflow ",
    "scikit-learn",
    "keras-rectified-adam",
    "numpy",
    "joblib",
]

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()



setup(
    name="proglearn",
    version="0.0.1",
    author="Will Levine, Jayanta Dey, Hayden Helm",
    author_email="levinewill@icloud.com",
    description="A package to implement and extend the methods desribed in 'A General Approach to Progressive Learning'",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/progressive-learning/",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=requirements,
)
