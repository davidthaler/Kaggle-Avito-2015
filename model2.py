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


def chunk_iterator(combo, chunk_size = 50000, start=0):
  '''
  Returns a generator that yields chunks of data.
  combo - the SFrame to draw from.
  '''
  global COL_NAMES
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


def select_train(chunk_it, all=True):
  '''
  A generator to filter rows from the other generator.
  The val/test/ds sets are now files.
  This filters train rows for full-train or all-data-train-val.
  '''
  for df in chunk_it:
    if all:                         # all labeled data incl. usual val set
      out = df[df.isTest == 0]
    else:                           # all labeled data except val set
      out = df[np.logical_and(df.isTest == 0, df.isVal == 0)]
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


def train(data, alpha=0.1, beta=1.0, L1=0.0, L2=0.1, D=2**26):
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


def predict(data, model, offset=0.0):
  out = []
  for k, df in enumerate(data):
    for t in df.itertuples():
      x, id = process_line(t, True)
      f = hash_features(x, model.D)
      dv = model.predict(f, False)
      dv += offset
      p = 1.0/(1.0 + exp(-dv))
      out.append((id, p))
  return pd.DataFrame(out, columns=['ID','IsClick'])
  
  
def write_submission(submit_id, preds):
  submit_name = 'submission%s.csv' % str(submit_id)
  submit_path = os.path.join(avito2_io.SUBMIT, submit_name)
  preds.to_csv(submit_path, index=False)

def val_run(tr, 
            val, 
            alpha=0.1, 
            beta=1.0, 
            L1=0.0, 
            L2=0.1, 
            D=2**26, 
            offset=-2.7):
  start = datetime.now()
  it = chunk_iterator(tr, chunk_size=50000)
  print 'training...'
  m = train(it, alpha=alpha, beta=beta, L1=L1, L2=L2, D=D)
  it = chunk_iterator(val, chunk_size=50000)
  print 'evaluating...'
  loss = validate(it, m, offset)
  print 'loss: %.5f' % loss
  print 'elapsed time: %s' % (datetime.now() - start)
  return m

def prepareSFrame(data):
  '''
  Call this on the SFrames before calling chunk_iterator, but after
  adding in extra.gl, if necessary.
  '''
  data['SearchQuery'] = data['SearchQuery'].apply(lambda s : s.strip().lower()[:5])
  data['ad_sq'] = data.apply(lambda x : x['SearchQuery'] + str(x['AdID']) if x['sqe'] else '_')
  data['st_pos'] = data['Position'] + 0.1 * data['seenToday']
  data['st_cat'] = data['CategoryID'] + 0.1 * data['seenToday']
  data['st_spe'] = data['spe'] + 0.1 * data['seenToday']
  data['st_sqe'] = data['sqe'] + 0.1 * data['seenToday']
  data['st_ad']  = data['AdID'] + 0.1 * data['seenToday']
  data['ad_pos'] = data['AdID'] + 0.1 * data['Position']
  data['ad_spe'] = data['AdID'] + 0.1 * data['spe']
  data['ad_sqe'] = data['AdID'] + 0.1 * data['sqe']
  

def full_train():
  combo = sframes.load('combo.gl')
  extras = sframes.load('extras.gl')
  combo.add_columms(extras)
  prepareSFrame(combo)
  cit = chunk_iterator(combo)
  sit = select_train(cit, True)
  model = train(sit)
  return model
  
def run_test(model):
  test = sframes.load('combo_test.gl')
  prepareSFrame(test)
  cit = chunk_iterator(test)
  pred = predict(cit, model)
  return pred

# Dead Code:

def extendSFrame(data):
  '''
  For now, call after prepareSFrame, and adding extras2.gl, is needed.
  Later, we'll merge this in.
  '''
  # ucat == user_clicked_ad_today
  data['ucat_pos'] = data['user_clicked_ad_today'] + 0.1 * data['Position']
  data['ucat_cat'] = data['user_clicked_ad_today'] + 0.1 * data['CategoryID']
  data['ucat_spe'] = data['user_clicked_ad_today'] + 0.1 * data['spe']
  data['ucat_sqe'] = data['user_clicked_ad_today'] + 0.1 * data['sqe']
  # NB: ucat_ad doesn't make sense
  
  # uc = user_clicked_today
  data['uc_pos'] = data['user_clicked_today'] + 0.1 * data['Position']
  data['uc_cat'] = data['user_clicked_today'] + 0.1 * data['CategoryID']
  data['uc_spe'] = data['user_clicked_today'] + 0.1 * data['spe']
  data['uc_sqe'] = data['user_clicked_today'] + 0.1 * data['sqe']
  data['uc_ad']  = data['user_clicked_today'] + 0.1 * data['AdID']
