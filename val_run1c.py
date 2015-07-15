'''
This script tests out downsampling of the negative rows and compensating
in the predictions by adjusting the offset. It does this naively, by pulling
all of the rows and rejecting the unused ones, so it is only slightly faster.
Its log-loss was higher by about 0.0001 when negatives were down sampled 95%.

NB: the most important line here is the offset calculation between the
training and validation passes.

author: David Thaler
date: July 2015
'''
import avito2_io
from hash_features import hash_features
from ftrl_proximal import ftrl_proximal
from datetime import datetime
from math import log, ceil, exp
from eval import logloss
import random
import pdb

alpha = 0.1        # learning rate
beta = 1.0         # smoothing parameter, probably doesn't matter on big data
L1 = 0.0000        # l1-regularization
L2 = 0.1000        # l2-regularization
D = 2**26          # feature space size
interaction = False
maxlines_train = None
maxlines_val = None
neg_ds = 0.05

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
              
ads_etl ={'price'  : lambda l : ceil(float(l[1])/100.),
          'ad_cat' : lambda l : l[0]}

# cut:
# 'params' : lambda l : len(l[3]),
# 'title'  : lambda l : len(l[2]),

# use_train = True
val_ids = avito2_io.get_artifact('full_val_set.pkl')
ads = avito2_io.get_artifact('context_ads.pkl')
print 'small objects loaded'
input = avito2_io.join_with_ads(True, 
                                ads,
                                train_etl, 
                                search_etl, 
                                ads_etl,
                                do_validation=False, 
                                val_ids=val_ids)
model = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# total count is just k + 1
total_y = 0.0
sample_ct = 0.0

for (k, (x, y)) in enumerate(input):
  total_y += y
  if y == 1 or random.random() < neg_ds:
    f = hash_features(x, D)
    p = model.predict(f)
    model.update(f, p, y)
    sample_ct += 1.0
  if k == maxlines_train:
    break
  if (k + 1) % 1000000 == 0:
    print 'processed %d lines on training pass' % (k + 1)
print 'finished training'

p_total =  total_y / (k + 1.0)
p_sample = total_y / sample_ct
offset = log(p_total/(1.0 - p_total)) - log(p_sample/(1.0 - p_sample))
print 'offset:' + str(offset)

# validation run
input = avito2_io.join_with_ads(True, 
                                ads,
                                train_etl, 
                                search_etl, 
                                ads_etl,
                                do_validation=True, 
                                val_ids=val_ids)
loss = 0.0
count = 0
for (k, (x, y)) in enumerate(input):
  f = hash_features(x, D)
  dv = model.predict(f, False)
  dv += offset
  p = 1.0/(1.0 + exp(-dv))
  loss += logloss(p, y)
  count += 1
  if k == maxlines_val:
    break
  if (k + 1) % 250000 == 0:
    print 'processed %d lines on validation pass' % (k + 1)
    
print 'validation set log loss: %.5f' % (loss/count)
print 'elapsed time: %s' % (datetime.now() - start)









