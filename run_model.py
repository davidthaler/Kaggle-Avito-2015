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
from eval import logloss
from datetime import datetime
from math import log, exp, ceil
import os
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
    diff = {'df_' + str(k):sp[k]+ap[k] for k in sp if k in ap and ap[k] != sp[k]}
    line.update(diff)
    line.update({('i_key' + str(k)) : 1 for k in i_keys})
    line.update({('i_kvs' + str(kv[0])) : kv[1] for kv in i_kvs})
    
  sq = line['SearchQuery']
  title = line['Title']
  if len(title) > MIN_LEN_STR and len(sq) > MIN_LEN_STR:
    line['sq_in_title'] = (title.lower().find(sq.lower()) > 0)

  line['SQexists'] = len(sq) > 0
  line['SPexists'] = sp is not None
  line['SPEcat'] = line['CategoryID'] + 0.1 * line['SPexists']
  line['SQEcat'] = line['CategoryID'] + 0.1 * line['SQexists']
  line['SPEad'] = line['AdID'] + 0.1 * line['SPexists']
  line['SQEad'] = line['AdID'] + 0.1 * line['SQexists']
  line['SQlen'] = round(log(1 + len(line['SearchQuery'])))
  line['Adlen'] = round(log(1 + len(line['Title'])))
  line['ad_pos'] = line['AdID'] + 0.1 * line['Position']
  line['cat_pos'] = line['CategoryID'] + 0.1 * line['Position']
  if 'UserAgentOSID' in line:
    line['osid_pos'] = line['UserAgentOSID'] + 0.1 * line['Position']
  if 'UserDeviceID' in line:
    line['udev_pos'] = line['UserDeviceID'] + 0.1 * line['Position']
  if 'UserAgentFamilyID' in line:
    line['ufam_pos'] = line['UserAgentFamilyID'] + 0.1 * line['Position']
    
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
    if (k + 1) % 250000 == 0:
      print 'processed %d lines on training pass' % (k + 1)
  return model


def compute_offset(tr, maxlines):
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
  
  if maxlines:
    n_click_tr = float(tr[:maxlines]['IsClick'].sum())
  else:
    n_click_tr = float(tr['IsClick'].sum())
  p_all = n_click_tr / TRAIN_ONLY_ROWS
  p_sample = n_click_tr / tr.shape[0]
  offset = log(p_all/(1.0 - p_all)) - log(p_sample/ (1.0 - p_sample))
  print 'Offset: %.5f' % offset
  return offset
  

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
    if (k + 1) % 250000 == 0:
      print 'processed %d lines from validation set' % (k + 1)
  return loss/k, k


def run_test(submission_file, test, si, offset):
  it = gl_iter.basic_join(test, si)
  for (k, line) in enumerate(it):
    id = line.pop('ID')
    process_line(line)
    f = hash_features(line, D)
    dv = model.predict(f, False)
    dv += offset
    p = 1.0/(1.0 + exp(-dv))
    submission_file.write('%d,%s\n' % (id, str(p)))
    if (k + 1) % 250000 == 0:
      print 'processed %d lines' % (k + 1)


if __name__ == '__main__':
  start = datetime.now()
  print 'running at: ' + str(start)
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
  offset = compute_offset(tr, args.maxlines)
  
  if args.sub:
    submit_name = 'submission%s.csv' % str(args.sub)
    submit_path = os.path.join(avito2_io.SUBMIT, submit_name)
    test = sframes.load('test_context.gl')
    si = sframes.load('search_test.gl')
    with open(submit_path, 'w') as sub_file:
      sub_file.write('ID,IsClick\n')
      run_test(sub_file, test, si, offset)
  else:
    val = sframes.load('val_context.gl')
    si  = sframes.load('search_val.gl')
    mean_loss, nrows = validate(val, si, offset, args.maxlines_val)
    print 'validation loss: %.5f on %d rows' % (mean_loss, nrows)
  print 'elapsed time: %s' % (datetime.now() - start)





