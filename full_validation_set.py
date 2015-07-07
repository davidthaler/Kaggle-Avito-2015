'''
This script records the steps for creating the object full_val_set.pkl 
in ARTIFACTS.

author: David Thaler
date: July 2015
'''
from avito2_io import *
from datetime import datetime

CUTOFF = '2015-05-08'

start = datetime.now()
print 'Reading test set ids...'
test_ids = get_artifact('test_search_ids.pkl')
print 'Collecting validation set ids...'
val_ids = full_val_set(test_ids, CUTOFF)
print 'Validation set size: %d' % len(val_ids)
print 'Writing validation set ids...'
put_artifact(val_ids, 'full_val_set.pkl')
print 'elapsed time: %s' % (datetime.now() - start)
