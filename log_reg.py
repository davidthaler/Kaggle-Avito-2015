'''
An initial logistic regression model for the Avito 2015 competition on Kaggle.

author: David Thaler
date: July 2015
'''
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


class LogRegModel(LogisticRegression):

  def score(self, x, y):
    probs = self.predict_proba(x)
    return log_loss(y, probs[:,1])
    