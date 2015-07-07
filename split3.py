'''
This script produces a train/val split for Avito2 (2015).
It is based on the 5% sample by value of UserID's in user3.pkl.
This split includes CategoryID, the day of week and hour, 
IsLoggedOn, SearchQuery exists, and SearchParams exists,
all extracted from SearchInfo.tsv.

author: David Thaler
date:July 2015
'''
from avito2_io import *
import os
from datetime import datetime

TEMPLATE = '{IsClick},{AdID},{HistCTR},{Position},{UserID},{CategoryID},{Weekday},'
TEMPLATE += '{Hour},{IsLoggedOn},{SQexists},{SPexists}\n'
HEADER = 'IsClick,AdID,HistCTR,Position,UserID,CategoryID,Weekday,Hour,'
HEADER += 'IsLoggedOn,SQexists,SPexists\n'
SPLIT_NUM = 3

start =  datetime.now()
sample = get_artifact('user3.pkl')
print 'loaded user sample dict'
test_sids = get_artifact('test_search_ids.pkl')
print 'loaded test set sids'
val_ids = extract_validation_ids(sample, test_sids)
print 'loaded val ids'

'''
* etl is a dict from field names to lambdas that extract and transform data from
* sample/user3.pkl, which comes out of SearchInfo.tsv
* NB: the form of sample/user3.pkl is: 
    {SearchID : [UserID, SearchDate, CategoryID, IsLoggedOn,
     SearchQuery exists, SearchParams exists]}
'''
# NB: I don't think it matters, but these values are ints, not str(int).
etl = {'UserID'     : (lambda t : t[0]),
       'CategoryID' : (lambda t : t[2]),
       'Weekday'    : (lambda t : t[1].weekday()),
       'Hour'       : (lambda t : t[1].hour),
       'IsLoggedOn' : (lambda t : t[3]),
       'SQexists'   : (lambda t : t[4]),
       'SPexists'   : (lambda t : t[5])}
       
lines = train_sample(sample, etl=etl)

trainpath = os.path.join(PROCESSED, 'train%d.csv' % SPLIT_NUM)
valpath   = os.path.join(PROCESSED, 'val%d.csv' % SPLIT_NUM)
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




