"""Small exact checks of historical-response estimators."""
import unittest
import numpy as np
import pandas as pd
from core import fit_local_response,fit_within_response,require_year

class EstimatorTests(unittest.TestCase):
    def test_year(self):
        with self.assertRaises(ValueError):require_year(pd.DataFrame({'date':['2025-01-01']}),2024)
    def test_scalar(self):
        self.assertAlmostEqual(fit_local_response([1.,2.,3.],[.25,.5,.75]),.25)
        self.assertEqual(fit_local_response([0.,0.],[1.,2.]),0.)
    def test_group_levels(self):
        x=np.array([[1.,0.],[2.,1.],[3.,0.],[4.,2.]])
        y=np.array([1.,3.,2.,4.]);g=np.array([1,1,2,2])
        shifted=x+np.where((g==1)[:,None],100.,-50.)
        self.assertTrue(np.allclose(fit_within_response(x,y,g),fit_within_response(shifted,y+np.where(g==1,8.,-6.),g)))

if __name__=='__main__':unittest.main()
