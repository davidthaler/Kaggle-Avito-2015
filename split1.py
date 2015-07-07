'''
This script produces the first train/val split for Avito2 (2015).
It is based on the 1% sample by value of UserID's in user2.pkl.

author: David Thaler
'''
from avito2_io import *
import os
from datetime import datetime

TEMPLATE = '{IsClick},{AdID},{HistCTR},{Position},{UserID}\n'
HEADER = 'IsClick,AdID,HistCTR,Position,UserID\n'
start =  datetime.now()
# sample has the form of user2: {SearchID: [UserID, SearchDate, CategoryID]}
sample = get_artifact('user2.pkl')
print 'loaded user sample dict'
test_sids = get_artifact('test_search_ids.pkl')
print 'loaded test set sids'
val_ids = extract_validation_ids(sample, test_sids)
print 'loaded val ids'
# Put sample into the form used by sample_train_by_user: {SearchID:UserID}
sample = {k:sample[k][0] for k in sample}
print 'rearranged sample'
lines = sample_train_by_user(-1, sample)

trainpath = os.path.join(PROCESSED, 'train1.csv')
valpath   = os.path.join(PROCESSED, 'val1.csv')
f_train = open(trainpath, 'w')
f_val   = open(valpath, 'w')
f_train.write(HEADER)
f_val.write(HEADER)
print 'begin reading train'
for (k, line) in enumerate(lines):
  if int(line['SearchID']) in val_ids:
    f_out = f_val
  else:
    f_out = f_train
  f_out.write(TEMPLATE.format(**line))
  if (k + 1) % 100000 == 0:
    print 'wrote %d lines' % (k + 1)
f_train.close()
f_val.close()
print 'elapsed time: %s' % (datetime.now() - start)



