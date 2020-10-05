..  -*- coding: utf-8 -*-

.. _contents:

Overview of ProgLearn_
===================

.. _proglearn: https://proglearn.neurodata.io/

.. image:: https://travis-ci.org/neurodata/ProgLearn.svg?branch=master
   :target: https://travis-ci.org/neurodata/progressive-learning
.. image:: https://codecov.io/gh/neurodata/ProgLearn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/neurodata/progressive-learning
.. image:: https://img.shields.io/pypi/v/proglearn.svg
   :target: https://pypi.org/project/proglearn/
.. image:: https://img.shields.io/badge/arXiv-2004.12908-red.svg?style=flat
   :target: https://arxiv.org/abs/2004.12908
.. image:: https://img.shields.io/badge/License-MIT-blue
   :target: https://opensource.org/licenses/MIT

``ProgLearn`` (**Prog**\ ressive **Learn**\ ing) is a package for exploring and using progressive learning algorithms.

Motivation
----------

In biological learning, data are used to improve performance simultaneously on the current task, as well as previouslyencountered and as yet unencountered tasks.  In contrast, classical machine learning starts from a blank slate,or tabula rasa, using data only for the single task at hand. While typical transfer learning algorithms can improveperformance on future tasks, their performance on prior tasks degrades upon learning new tasks (called catastrophicforgetting).  Many recent approaches have attempted tomaintainperformance given new tasks.  But striving toavoid forgetting sets the goal unnecessarily low: the goal of ``ProgLearn`` is to improveperformance on all tasks (including past and future) with any new data.

Python
------

Python is a powerful programming language that allows concise expressions of
network algorithms.  Python has a vibrant and growing ecosystem of packages
that hyppo uses to provide more features such as numerical linear algebra and
plotting.  In order to make the most out of ``ProgLearn`` you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

``ProgLearn`` is free software; you can redistribute it and/or modify it under the
terms of the :doc:`MIT </license>`.  We welcome contributions. Join us on
`GitHub <https://github.com/neurodata/proglearn>`_.

History
-------

``ProgLearn`` was founded in February 2020. The source code was designed and written by 
Will LeVine and Hayden Helm. The experiment code was designed and written by Jayanta Dey 
and Will LeVine. The repository is maintained by Will LeVine and Jayanta Dey. 

Documentation
=============

.. toctree::
   :maxdepth: 1

   install
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
