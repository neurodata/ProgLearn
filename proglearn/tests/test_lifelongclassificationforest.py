import unittest
import pytest
import numpy as np
import random

from proglearn.forest import LifelongClassificationForest
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage

class test_LifelongClassificationForest(unittest.TestCase):
    
    def setUp(self):
        self.l2f = LifelongClassificationForest()
    
    def test_initialize(self):
        self.assertTrue(True)
        
    def test_correct_default_transformer(self):
        self.assertIs(self.l2f.pl.default_transformer_class, TreeClassificationTransformer)
        
    def test_correct_default_voter(self):
        self.assertIs(self.l2f.pl.default_voter_class, TreeClassificationVoter)
        
    def test_correct_default_decider(self):
        self.assertIs(self.l2f.pl.default_decider_class, SimpleArgmaxAverage)
        
    def test_correct_default_kwargs_transformer_decider_empty(self):
        self.assertFalse(self.l2f.pl.default_transformer_kwargs)
        self.assertFalse(self.l2f.pl.default_decider_kwargs)
        
    def test_correct_default_estimators(self):
        self.assertIs(self.l2f.n_estimators, 100)
        
    def test_correct_estimator(self):
        rand = random.randint(0, 100)
        l2f = LifelongClassificationForest(n_estimators=rand)
        self.assertIs(l2f.n_estimators, rand)
        
    def test_correct_default_finite_sample_correction(self):
        tmp_dict = {"finite_sample_correction": False}
        self.assertEqual(self.l2f.pl.default_voter_kwargs, tmp_dict)
    
    def test_correct_true_initilization_finite_sample_correction(self):
        tmp_dict = {"finite_sample_correction": True}
        l2f = LifelongClassificationForest(finite_sample_correction=True)
        self.assertEqual(l2f.pl.default_voter_kwargs, tmp_dict)
        
if __name__ == '__main__':
    unittest.main()