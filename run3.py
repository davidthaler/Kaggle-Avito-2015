'''
This script runs the full-data run of the ftrl-proximal model using 
data from gl_features.features2(). This only runs one epoch (~200M rows).
It differs a little from run2.py in that the features2 data contains fields
that have to be removed (SearchID and SearchDate).

author: David Thaler
date: July 2015
'''
from avito2_io import SUBMIT, PROCESSED
from hash_features import hash_features
from ftrl_proximal import ftrl_proximal
from datetime import datetime
from math import log
import os.path
import csv
import pdb

SUBMIT_NUM = 3
submission = os.path.join(SUBMIT, 'submission%d.csv' % SUBMIT_NUM)
TRAIN_INFILE = 'gl_train2.csv'
TEST_INFILE = 'gl_test2.csv'


alpha = 0.1       # learning rate
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
L1 = 0.0000        # l1-regularization
L2 = 0.1000        # l2-regularization
D = 2**26          # feature space size
interaction = False
maxlines = None

start = datetime.now()
model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
train_path = os.path.join(PROCESSED, TRAIN_INFILE)
with open(train_path) as train_file:
  input = csv.DictReader(train_file)
  for (k, x) in enumerate(input):
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


outfile = open(submission, 'w')
outfile.write('ID,IsClick\n')
test_path = os.path.join(PROCESSED, TEST_INFILE)
with open(test_path) as test_file:
  input = csv.DictReader(test_file)
  for (k, x) in enumerate(input):
    id = x['ID']
    del x['ID']
    del x['SearchDate']
    f = hash_features(x, D)
    p = model.predict(f)
    outfile.write('%s,%s\n' % (id, str(p)))
    if (k + 1) % 1000000 == 0:
      print 'processed %d lines' % (k + 1)
print 'elapsed time: %s' % (datetime.now() - start)









