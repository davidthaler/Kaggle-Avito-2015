'''
This script implements a full-data run of the ftrl-proximal model using 
data pulled by avito2_io.rolling_join(), with validation (no test).
This only runs one epoch (~200M rows).

author: David Thaler
date: July 2015
'''
import avito2_io
from hash_features import hash_features
from ftrl_proximal import ftrl_proximal
from datetime import datetime
from math import log
from eval import logloss


alpha = 0.1        # learning rate
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
L1 = 0.0000        # l1-regularization
L2 = 0.1000        # l2-regularization
D = 2**26          # feature space size
interaction = False
maxlines_train = None
maxlines_val = None

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
val_ids = avito2_io.get_artifact('full_val_set.pkl')
input = avito2_io.rolling_join(True, 
                               train_etl, 
                               search_etl, 
                               do_validation=False, 
                               val_ids=val_ids)
model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

for (k, (x, y)) in enumerate(input):
  f = hash_features(x, D)
  p = model.predict(f)
  model.update(f, p, y)
  if k == maxlines_train:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines on training pass' % (k + 1)
print 'finished training'

# validation run
input = avito2_io.rolling_join(True, 
                               train_etl, 
                               search_etl, 
                               do_validation=True, 
                               val_ids=val_ids)
loss = 0.0
count = 0
for (k, (x, y)) in enumerate(input):
  f = hash_features(x, D)
  p = model.predict(f)
  loss += logloss(p, y)
  count += 1
  if k == maxlines_val:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines on validation pass' % (k + 1)
    
print 'validation set log loss: %.5f' % (loss/count)
print 'elapsed time: %s' % (datetime.now() - start)









