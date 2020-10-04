# ProgLearn

[![Build Status](https://travis-ci.org/neurodata/progressive-learning.svg?branch=master)](https://travis-ci.org/neurodata/progressive-learning)
[![codecov](https://codecov.io/gh/neurodata/progressive-learning/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/progressive-learning)
[![PyPI version](https://img.shields.io/pypi/v/proglearn.svg)](https://pypi.org/project/proglearn/)
[![arXiv shield](https://img.shields.io/badge/arXiv-2004.12908-red.svg?style=flat)](https://arxiv.org/abs/2004.12908)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

`proglearn` (**Prog**ressive **Learn**ing) is a package for exploring and using progressive learning algorithms developed by the [neurodata group](https://neurodata.io).

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Contributing](#contributing)
- [License](#license)
- [Issues](#issues)

# Overview
The natural process of biological learning involves progressive acquisition of new information developing on past knowledge and experiences, which often leads to a performance improvement on a given task. Learning a second language, for instance, is associated with higher performance in an individual’s native language compared to that of monolinguals. In classical machine learning, the process usually begins from the state of <i>tabula rasa</i>, zero knowledge, and is optimized for a single task. The issues arise when the system is sequentially optimized for multiple tasks exhibiting “catastrophic forgetting,” diminishing performance of previously learned tasks. One of the current limitations of artificial intelligence revolves around this inability to transfer knowledge. <br><br>
The progressive learning package utilizes representation ensembling algorithms to sequentially learn a representation for each task and ensemble both old and new representations for all future decisions. Here, two complementary representation ensembling algorithms based on decision forests (Lifelong Forest) and deep networks (Lifelong Network) demonstrate forward and backward knowledge transfer of tasks on multiple real datasets, including both vision and language applications.

# Documentation


# System Requirements
## Hardware requirements
`proglearn` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux* and *macOS*. The package has been tested on the following systems:
+ Linux: Ubuntu 16.04
+ macOS: Mojave (10.14.1)
+ Windows: 10

### Python Requirements
This package is written for Python3. Currently, it is supported for Python 3.6 and 3.7.

### Python Dependencies
`proglearn` mainly depends on the Python scientific stack.
```
keras>=2.3.1
tensorflow>=1.19.0
scikit-learn>=0.22.0
scipy==1.4.1
numpy<1.19
joblib>=0.14.1
```

# Installation Guide
## Install from pip
```
pip install proglearn
```

## Install from Github
```
git clone https://github.com/neurodata/ProgLearn.git
cd ProgLearn
python3 setup.py install
```

# Contributing
We welcome contributions from anyone. Please see our [contribution guidelines](https://github.com/neurodata/ProgLearn/blob/master/CONTRIBUTING.md) before making a pull request. Our
[issues](https://github.com/neurodata/ProgLearn/issues) page is full of places we could use help!
If you have an idea for an improvement not listed there, please
[make an issue](https://github.com/neurodata/ProgLearn/issues/new) first so you can discuss with the
developers.

# License
This project is covered under the [MIT License](https://github.com/neurodata/ProgLearn/blob/master/LICENSE).

# Issues
We appreciate detailed bug reports and feature requests (though we appreciate pull requests even more!). Please visit our [issues](https://github.com/neurodata/ProgLearn/issues) page if you have questions or ideas.

# Citing ProgLearn
If you find ProgLearn useful in your work, please cite the package via the [progressive-learning paper](https://arxiv.org/pdf/2004.12908.pdf)

> Vogelstein JT, Helm HS, Mehta RD, Dey J, Yang W, Tower B, LeVine W, Larson J, White C, Priebe CE. A general approach to progressive learning. arXiv preprint arXiv:2004.12908. 2020 Apr 27.
