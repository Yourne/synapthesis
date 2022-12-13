#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:14:35 2022

@author: nepal
"""

from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np


def negloglike(params, x) -> float:
    mu, sigma = params
    return -np.sum(norm.logpdf(x, loc=mu, scale=sigma))


rng = np.random.default_rng(seed=42)

sample = rng.normal(loc=7, scale=3, size=20)

model = minimize(negloglike, [0, 1], args=(sample), method="L-BFGS-B")

    


    
