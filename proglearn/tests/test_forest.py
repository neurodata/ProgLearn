import unittest
import pytest
import numpy as np
import random

from proglearn.forest import LifelongClassificationForest
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage

class TestLifelongClassificationForest:
    
    def test_initialize(self):
        l2f = LifelongClassificationForest()
        assert True
        
    def test_correct_default_transformer(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl.default_transformer_class == TreeClassificationTransformer
        
    def test_correct_default_voter(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl.default_voter_class == TreeClassificationVoter
        
    def test_correct_default_decider(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl.default_decider_class == SimpleArgmaxAverage
        
    def test_correct_default_kwargs(self):
        l2f = LifelongClassificationForest()
        
        #transformer 
        assert l2f.pl.default_transformer_kwargs == {}
        
        #voter
        assert len(l2f.pl.default_voter_kwargs) == 1
        assert "finite_sample_correction" in list(l2f.pl.default_voter_kwargs.keys())
        assert l2f.pl.default_voter_kwargs["finite_sample_correction"] == False
        
        #decider
        assert l2f.pl.default_decider_kwargs == {}
        
    def test_correct_default_n_estimators(self):
        l2f = LifelongClassificationForest()
        assert l2f.n_estimators == 100
    
    def test_correct_true_initilization_finite_sample_correction(self):
        l2f = LifelongClassificationForest(default_finite_sample_correction=True)
        assert l2f.pl.default_voter_kwargs == {"finite_sample_correction": True}
