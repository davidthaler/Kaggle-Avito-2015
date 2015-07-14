'''
This script runs the full-data run of the ftrl-proximal model using 
data from gl_features.features2(). This only runs one epoch (~200M rows).
It differs a little from run2.py in that the features2 data contains fields
that have to be removed (SearchID and SearchDate).

author: David Thaler
date: July 2015
'''
import avito2_io
from hash_features import hash_features
from ftrl_proximal import ftrl_proximal
from datetime import datetime
from math import log
from eval import logloss
import os.path
import csv
import argparse
import pdb


TRAIN_INFILE = 'gl_train2.csv'
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
D = 2**26          # feature space size

def run_val(alpha, l2, l1, maxlines, interact):
  val_ids = avito2_io.get_artifact('full_val_set.pkl')
  model = ftrl_proximal(alpha, beta, l1, l2, D, interact)
  train_path = os.path.join(avito2_io.PROCESSED, TRAIN_INFILE)
  with open(train_path) as train_file:
    input = csv.DictReader(train_file)
    for (k, x) in enumerate(input):
      if int(x['SearchID']) not in val_ids:
        y = float(x['IsClick'])
        del x['IsClick']
        del x['SearchDate']
        del x['SearchID']
        f = hash_features(x, D)
        p = model.predict(f)
        model.update(f, p, y)
      if k == maxlines:
        break
      if (k + 1) % 1000000 == 0:
        print 'processed %d lines' % (k + 1)
  print 'finished training'
  count = 0
  loss = 0.0
  with open(train_path) as train_file:
    input = csv.DictReader(train_file)
    for (k, x) in enumerate(input):
      if int(x['SearchID']) in val_ids:
        count += 1
        y = float(x['IsClick'])
        del x['IsClick']
        del x['SearchDate']
        del x['SearchID']
        f = hash_features(x, D)
        p = model.predict(f)
        loss += logloss(p, y)
      if k == maxlines:
        break
      if (k + 1) % 1000000 == 0:
        print 'processed %d lines of raw train on validation pass' % (k + 1)
  print 'validation loss: %.5f on %d rows' % (loss/count, count)


if __name__=='__main__':
  start = datetime.now()
  parser = argparse.ArgumentParser(description=
            'Trains with given parameters, then computes validation log-loss.')
  parser.add_argument('--alpha', type=float, default=0.1,
                      help='initial learning rate')
  parser.add_argument('--l2', type=float, default=0.1, 
                      help='L2 regularization strength')
  parser.add_argument('--l1', type=float, default=0.0,
                      help='L1 regularization strength')
  parser.add_argument('-m', '--maxlines',type=int, default=None,
                      help='A max # of lines to use, if none, all data is used.')
  parser.add_argument('-i', '--interact', action='store_const', 
                       const=True, default=False,
                       help='Create interaction features for all pairs of values')
  args = parser.parse_args()
  run_val(args.alpha, args.l2, args.l1, args.maxlines, args.interact)
  print 'elapsed time: %s' % (datetime.now() - start)




