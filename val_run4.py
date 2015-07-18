'''
This script runs the ftrl-proximal model on data from gl_iter.basic_join.

author: David Thaler
date: July 2015
'''
import avito2_io
import gl_iter
import sframes
from ftrl_proximal import ftrl_proximal
from hash_features import hash_features
from datetime import datetime
from eval import logloss
from math import log, exp, ceil
import pdb


alpha = 0.1        # learning rate
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
L1 = 0.0001        # l1-regularization
L2 = 0.0010        # l2-regularization
D = 2**26          # feature space size
interaction = False
maxlines_train = None
maxlines_val = None
MIN_LEN = 4
# This is (# rows in train_context) - (# rows in val_context).
# It is the # of rows train_ds was drawn from.
TRAIN_ONLY_ROWS = 184967172.0

def process_line(line):
  '''
  NB: We are modifying line by reference here.
  '''
  del line['SearchID']
  del line['SearchDate']
  line['HistCTR'] = -10 * round(log(float(line['HistCTR'])), 1)
  line['Price'] =  ceil(float(line['Price'])/100.)
  ap = line['Params']
  if ap is not None:
    ad_keys = {('ad_key' + str(k)) : 1 for k in ap}
    ad_kvs  = {('ad_kvs' + str(k)) : ap[k] for k in ap}
    line.update(ad_keys)
    line.update(ad_kvs)
  sp = line['SearchParams']
  if sp is not None:
    sp_keys = {('sp_key' + str(k)) : 1 for k in sp}
    sp_kvs  = {('sp_kvs' + str(k)) : sp[k] for k in sp}
    line.update(sp_keys)
    line.update(sp_kvs)
  if ap is not None and sp is not None:
    i_keys = set(ap.keys()).intersection(set(sp.keys()))
    i_kvs  = set(ap.items()).intersection(set(ap.items()))
    line.update({('i_key' + str(k)) : 1 for k in i_keys})
    line.update({('i_kvs' + str(kv[0])) : kv[1] for kv in i_kvs})
    
  sq = line['SearchQuery']
  title = line['Title']
  if len(title) > MIN_LEN and len(sq) > MIN_LEN:
    line['sq_in_title'] = (title.find(sq) > 0)
    
  # These have been unpacked already
  del line['Params']
  del line['SearchParams']
  
start = datetime.now()
tr = sframes.load('train_ds.gl')
si = sframes.load('search_ds.gl')
it = gl_iter.basic_join(tr, si)
model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
for (k, line) in enumerate(it):
  y = line.pop('IsClick')
  process_line(line)
  f = hash_features(line, D)
  p = model.predict(f)
  model.update(f, p, y)
  if k == maxlines_train:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines on training pass' % (k + 1)
print 'finished training'

# Sample bias adjustment
n_click_tr = float(tr['IsClick'].sum())
p_all = n_click_tr / TRAIN_ONLY_ROWS
p_sample = n_click_tr / tr.shape[0]
offset = log(p_all/(1.0 - p_all)) - log(p_sample/ (1.0 - p_sample))

val = sframes.load('val_context.gl')
si  = sframes.load('search_val.gl')
it = gl_iter.basic_join(val, si)
loss = 0.0
for (k, line) in enumerate(it):
  y = line.pop('IsClick')
  process_line(line)
  f = hash_features(line, D)
  dv = model.predict(f, False)
  dv += offset
  p = 1.0/(1.0 + exp(-dv))
  loss += logloss(p, y)
  if k == maxlines_val:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines from validation set' % (k + 1)

print 'validation loss: %.5f on %d rows' % (loss/k, k)
print 'elapsed time: %s' % (datetime.now() - start)








