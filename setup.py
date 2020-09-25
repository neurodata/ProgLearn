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
    author="Will Levine, Jayanta Dey, Hayden Helm",
    author_email="levinewill@icloud.com",
    description="A package to implement and extend the methods desribed in 'A General Approach to Progressive Learning'",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neurodata/progressive-learning/",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
)
