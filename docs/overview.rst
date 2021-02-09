..  -*- coding: utf-8 -*-

.. _contents:

Overview of ProgLearn_
======================

.. _proglearn: https://proglearn.neurodata.io/

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4276573.svg
   :target: https://doi.org/10.5281/zenodo.4276573
.. image:: https://travis-ci.org/neurodata/ProgLearn.svg?branch=main
   :target: https://travis-ci.org/neurodata/ProgLearn
.. image:: https://codecov.io/gh/neurodata/ProgLearn/branches/main/graph/badge.svg
   :target: https://codecov.io/gh/neurodata/ProgLearn
.. image:: https://img.shields.io/pypi/v/proglearn.svg
   :target: https://pypi.org/project/proglearn/
.. image:: https://img.shields.io/badge/arXiv-2004.12908-red.svg?style=flat
   :target: https://arxiv.org/abs/2004.12908
.. image:: https://img.shields.io/badge/License-MIT-blue
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/netlify/97f86f49-81ed-4292-a100-f7031b54ecc7
   :target: https://app.netlify.com/sites/neuro-data-proglearn/deploys
.. image:: https://img.shields.io/pypi/dm/proglearn.svg
   :target: https://pypi.org/project/proglearn/#files

``ProgLearn`` (**Prog**\ ressive **Learn**\ ing) is a package for exploring and using progressive learning algorithms.

``ProgLearn`` provides classes and functions for biological machine learning. Notably, it improves in performance on all tasks (including past and future) with any new data. This sets it apart from classical machine learning algorithms and many other recent approaches to biological learning.

The Library
----------

All classes and functions are available through the ``ProgLearn`` package and can also be imported separately.

``import ProgLearn as PL``

``from proglearn.forest import UncertaintyForest``

The Learners
------

There are three main parts to the ``ProgLearn`` package: Lifelong Classification Network, Lifelong Classification Forest, and the Uncertainty Forest. All three have very similar syntax and usage. A general overview is provided below with more specific and complete examples in the tutorial section. This overview example will use ``ProgLearn.forest.UncertaintyForest`` but is generalizable to the Lifelong Classification Network and Lifelong Classification Forest.

First, we'll create our forest:

``UF = UncertaintyForest(n_estimators = n_estimators)``

Then, we fit to data:

``UF.fit(X_train, y_train)``

Finally we can predict the classes of the data:

``predictions = UF.predict(X_test)``

Another advantage of the ProgLearn package is the ``predict_proba(X)`` method. It can be used to estimate the class posteriors for each example in the input data, X.

``predicted_posteriors = UF.predict_proba(X_test)``


Wrap Up
-------------

This covers the basics of using ProgLearn. Most use cases will use the methods presented above. Further examples are available in the tutorials section (see links below).

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Using ProgLearn

   overview
   install
   tutorials/
   reference/index
   contributing
   license

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   proglearn @ GitHub <https://github.com/neurodata/proglearn>
   proglearn @ PyPi <https://pypi.org/project/proglearn/>
   Issue Tracker <https://github.com/neurodata/proglearn/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
