..  -*- coding: utf-8 -*-

.. _contents:

Overview
========

``ProgLearn`` provides classes and functions for biological machine learning. Notably, it improves in performance on all tasks (including past and future) with any new data. This sets it apart from classical machine learning algorithms and many other recent approaches to biological learning.

The Library
-----------

All classes and functions are available through the ``ProgLearn`` package and can also be imported separately.

.. code:: python

    import proglearn as PL
    from proglearn.forest import UncertaintyForest

The Learners
------------

There are three main parts to the ``ProgLearn`` package: Lifelong Classification Network, Lifelong Classification Forest, and the Uncertainty Forest. All three have very similar syntax and usage. A general overview is provided below with more specific and complete examples in the tutorial section. This overview example will use ``proglearn.forest.UncertaintyForest`` but is generalizable to the Lifelong Classification Network and Lifelong Classification Forest.

First, we'll create our forest:

.. code:: python

    UF = UncertaintyForest(n_estimators = n_estimators)

Then, we fit to data:

.. code:: python

    UF.fit(X_train, y_train)

Finally we can predict the classes of the data:

.. code:: python

    predictions = UF.predict(X_test)

Another advantage of the ``ProgLearn`` package is the ``predict_proba(X)`` function. It can be used to estimate the class posteriors for each example in the input data, X.

.. code:: python

  predicted_posteriors = UF.predict_proba(X_test)


Wrap Up
-------

This overview covers the basics of using ``ProgLearn``. Most use cases will utilize the functions presented above. Further examples are available in the tutorials section (see menu).
