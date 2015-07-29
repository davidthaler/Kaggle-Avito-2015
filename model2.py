'''
This script runs the ftrl-proximal model on data from combo.gl and extras.gl.

author: David Thaler
date: July 2015
'''
import avito2_io
from run_model import compute_offset
import sframes
from ftrl_proximal import ftrl_proximal
from hash_features import hash_features
from eval import logloss
from datetime import datetime
from math import log, exp, ceil
import os
import argparse
import numpy as np
import pandas as pd
import pdb

DROP_COLS = ['rowId','SearchID', 'isDS', 'isVal', 'isTest', 'IsClick','ID']
ROUND_COLS = ['HistCTR', 'cat_pos', 'spe_cat','spe_pos','sqe_cat','sqe_pos']
COL_NAMES = None
VAL_START = 0
TEST_START = 0

def chunk_iterator(chunk_size = 10000, start=0):
  '''
  Returns a generator that yields chunks of data.
  '''
  global COL_NAMES
  combo = sframes.load('combo.gl')
  extras = sframes.load('extras.gl')
  combo.add_columns(extras)
  names = combo.column_names()
  names.insert(0, 'rowId')
  COL_NAMES = names
  chunk_start = start
  data_end = combo.shape[0]
  chunk_end = min(data_end, start + chunk_size)
  while chunk_start < data_end:
    print 'chunk [%d:%d]' % (chunk_start, chunk_end)
    chunk = combo[chunk_start:chunk_end].to_dataframe()
    chunk_start += chunk_size
    chunk_end = min(data_end, chunk_end + chunk_size)
    yield chunk


def select_it(chunk_it, what, maxlines=None):
  nrows = 0
  for df in chunk_it:
    if what == 'ds':              # downsampled train only (not val)
      out = df[df.isDS == 1]
    elif what == 'val':           # val set
      out = df[df.isVal == 1]
    elif what == 'full_train':    # all labeled data incl. usual val set
      out = df[df.isTest == 0]
    elif what == 'test':          # test set
      out = df[df.isTest == 1]
    elif what == 'train_val':     # all labeled data except val set
      out = df[np.logical_and(df.isTest == 0, df.isVal == 0)]
    nrows += out.shape[0]
    if maxlines is not None and nrows >= maxlines:
      break
    if out.shape[0] > 0:
      yield out
    
def process_line(t, isTest):
  line = dict(zip(COL_NAMES, t))
  if isTest:
    y = line['ID']
  else:
    y = line['IsClick']
  for col in DROP_COLS:
    line.pop(col)
  for col in ROUND_COLS:
    line[col] = round(line[col], 1)
  line.pop('SearchParams')
  line.pop('dt')
  return line, y


def train(data, alpha, beta, L1, L2, D):
  '''
  Runs one training pass.
  '''
  model = ftrl_proximal(alpha, beta, L1, L2, D, False)
  for df in data:
    for t in df.itertuples():
      x, y = process_line(t, False)
      f = hash_features(x, D)
      p = model.predict(f)
      model.update(f, p, y)
  return model


def validate(data, model, offset=0.0):
  loss = 0.0
  count = 0
  for k, df in enumerate(data):
    for t in df.itertuples():
      count += 1
      x, y = process_line(t, False)
      f = hash_features(x, model.D)
      dv = model.predict(f, False)
      dv += offset
      p = 1.0/(1.0 + exp(-dv))
      loss += logloss(p, y)
  return loss/count



