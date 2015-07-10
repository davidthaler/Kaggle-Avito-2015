'''
This script records the generation of the first full-data run of the 
ftrl-proximal model using data pulled by avito2_io.rolling_join().
This only runs one epoch (~200M rows).

author: David Thaler
date: July 2015
'''
from avito2_io import rolling_join
from avito2_io import SUBMIT
from hash_features import hash_features
from ftrl_proximal import ftrl_proximal
from datetime import datetime
from math import log
from eval import logloss
import os.path
import pdb

SUBMIT_NUM = 1
submission = os.path.join(SUBMIT, 'submission%d.csv' % SUBMIT_NUM)

alpha = 0.1       # learning rate
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
L1 = 0.0000        # l1-regularization
L2 = 0.1000        # l2-regularization
D = 2**26          # feature space size
interaction = False
maxlines = None


start = datetime.now()
train_etl = {'ad'     : (lambda l : l['AdID']),
             'pos'    : (lambda l : l['Position']),
             'log_ctr': (lambda l : -10 * round(log(float(l['HistCTR'])), 1))}
search_etl = {'user'    : (lambda l : l['UserID']),
              'category': (lambda l : l['CategoryID']),
              'location': (lambda l : l['LocationID']),
              'logon'   : (lambda l : l['IsUserLoggedOn']),
              'SPexists': (lambda l : int(len(l['SearchParams']) > 0)),
              'SQexists': (lambda l : int(len(l['SearchQuery']) > 0))}
# use_train = True
input = rolling_join(True, train_etl, search_etl)
model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
for (k, (x, y)) in enumerate(input):
  f = hash_features(x, D)
  p = model.predict(f)
  model.update(f, p, y)
  if k == maxlines:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines' % (k + 1)
print 'finished training'

# testing: use_train=False
train_etl['id'] = (lambda l : l['ID'])
input = rolling_join(False, train_etl, search_etl)
outfile = open(submission, 'w')
outfile.write('ID,IsClick\n')
# y is just 0 here, and it isn't used
for (k, (x, y)) in enumerate(input):
  f = hash_features(x, D)
  p = model.predict(f)
  outfile.write('%s,%s\n' % (x['id'], str(p)))
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines' % (k + 1)
print 'elapsed time: %s' % (datetime.now() - start)











