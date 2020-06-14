from setuptools import setup, find_packages

requirements = [
    "keras",
    "tensorflow ",
    "scikit-learn",
    "keras-rectified-adam",
    "numpy",
    "joblib",
]

# Uncomment once README.md exists

# with open("README.md", "r") as readme_file:
#     readme = readme_file.read()



setup(
    name="proglearn",
    version="0.0.1",
    author="Will Levine, Jayanta Dey, Hayden Helm",
    author_email="levinewill@icloud.com",
    description="A package to implement and extend the methods desribed in 'A General Approach to Progressive Learning'",
    # long_description=readme,
    # long_description_content_type="text/markdown",
    url="https://github.com/neurodata/progressive-learning/",
    packages=find_packages(),
    install_requires=requirements,
)