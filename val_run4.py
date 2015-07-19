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
import argparse
import pdb


def process_line(line):
  '''
  NB: We are modifying line by reference here.
  '''
  MIN_LEN_STR = 4
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
  if len(title) > MIN_LEN_STR and len(sq) > MIN_LEN_STR:
    line['sq_in_title'] = (title.find(sq) > 0)
    
  # These have been unpacked already
  del line['Params']
  del line['SearchParams']
  
def train(tr, si, alpha, beta, L1, L2, D, interaction, maxlines):
  it = gl_iter.basic_join(tr, si)
  model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
  for (k, line) in enumerate(it):
    y = line.pop('IsClick')
    process_line(line)
    f = hash_features(line, D)
    p = model.predict(f)
    model.update(f, p, y)
    if k == maxlines:
      break
    if (k + 1) % 1000000 == 0:
      print 'processed %d lines on training pass' % (k + 1)
  return model


def compute_offset(tr):
  '''
  Using down sampled negative biases the mean prediction. This function
  computes an adjustment to correct that bias.
  
  args:
    tr - the data with negative instances down-sampled
    
  return:
    the offset to add to decision values to compensate sample bias
  '''
  # This is (# rows in train_context) - (# rows in val_context).
  # It is the # of rows train_ds was down-sampled from.
  TRAIN_ONLY_ROWS = 184967172.0
  
  n_click_tr = float(tr['IsClick'].sum())
  p_all = n_click_tr / TRAIN_ONLY_ROWS
  p_sample = n_click_tr / tr.shape[0]
  return log(p_all/(1.0 - p_all)) - log(p_sample/ (1.0 - p_sample))
  

def validate(val, si, offset, maxlines):
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
    if k == maxlines:
      break
    if (k + 1) % 1000000 == 0:
      print 'processed %d lines from validation set' % (k + 1)
  return loss/k, k


if __name__ == '__main__':
  start = datetime.now()
  parser = argparse.ArgumentParser(description=
        'Runs ftrl-proximal model on data from gl_iter.basic_join')
  parser.add_argument('--alpha', type=float, default=0.1,
                      help='initial learning rate')
  parser.add_argument('--beta', type=float, default=1.0,
                      help='smoothing parameter')
  parser.add_argument('--l2', type=float, default=0.1, 
                      help='L2 regularization strength')
  parser.add_argument('--l1', type=float, default=0.0,
                      help='L1 regularization strength')
  parser.add_argument('-b', '--bits', type=int, default=26,
                      help='use 2**bits feature space dimension')
  parser.add_argument('-m', '--maxlines',type=int, default=None,
        help='A max # lines to use for train, if none, all data is used.')
  parser.add_argument('-n', '--maxlines_val',type=int, default=None,
        help='A max # lines for validation, if none, all data is used.')
  parser.add_argument('-s', '--sub', type=str,
        help='None or str. If given, predict on test and write submission.')
  args = parser.parse_args()
  D = 2**args.bits
  # for now, we're just using *ds, because full data will take too long
  tr = sframes.load('train_ds.gl')
  si = sframes.load('search_ds.gl') 
  # no interactions; it'd take days
  model = train(tr, 
                si, 
                args.alpha, 
                args.beta, 
                args.l1, 
                args.l2, 
                D, 
                False, 
                args.maxlines)
  print 'finished training'
  offset = compute_offset(tr)
  
  # for now, just to get things running, we'll just validate
  val = sframes.load('val_context.gl')
  si  = sframes.load('search_val.gl')
  mean_loss, nrows = validate(val, si, offset, args.maxlines_val)
  print 'validation loss: %.5f on %d rows' % (mean_loss, nrows)
  print 'elapsed time: %s' % (datetime.now() - start)





