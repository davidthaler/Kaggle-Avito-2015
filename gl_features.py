import graphlab as gl
import sframes
import avito2_io
import os
from math import log
from datetime import datetime


def get_context_ads():
  '''
  Loads the ads.gl file (all of AdsInfo.tsv) and extracts the context ads.
  '''
  ads = sframes.load('ads.gl')
  ctx = ads[ads['IsContext']]
  del ctx['IsContext']
  del ctx['Location']
  return ctx

def features1():
  '''
  This function implements and records the construction of the first sframe
  features, which are the same as the features used in the pure python/pypy 
  run1.py. This uses just SearchInfo and trainSearchStream and runs row-wise
  on context ads only.
  '''
  # process trainSearchStream
  start = datetime.now()
  print 'ingesting train.gl'
  tr = sframes.load('train.gl')
  tr = tr[tr['ObjectType'] == 3]
  del tr['ObjectType']
  tr['log_ctr'] = tr['HistCTR'].apply(lambda x : -10 * round(log(x), 1))
  del tr['HistCTR']
  print 'train.gl ingested, elapsed time: %s' % (datetime.now() - start)
  
  # process SearchInfo
  print 'ingesting search.gl'
  si = sframes.load('search.gl')
  # In run1.py, we didn't use date or IPID
  del si['SearchDate']
  del si['IPID']
  si['SQexists'] = si['SearchQuery'].apply(lambda x : len(x) > 0)
  del si['SearchQuery']
  # NB: lambda d : 0 if d is None else len(d) doesn't seem to work
  si['SPexists'] = (si['SearchParams'].apply(lambda d : int(d is not None))
                                      .fillna(0))
  del si['SearchParams']
  print 'search.gl ingested, elapsed time: %s' % (datetime.now() - start)
  
  # join up training set
  # NB: due to lazy evaluation, this might not time accurately
  print 'joining up training set...'
  f = tr.join(si, how='left', on='SearchID')
  # This line makes validation impossible. Run script must delete SearchID.
  #del f['SearchID']
  print 'join completed, elapsed time: %s' % (datetime.now() - start)
  
  # save training features
  path = os.path.join(avito2_io.PROCESSED, 'gl_train1.csv')
  print 'saving training features to %s' % path
  f.save(path, format='csv')
  print 'training features saved, elapsed time: %s' % (datetime.now() - start)
  
  # load test set
  print 'ingesting test.gl'
  test = sframes.load('test.gl')
  test = test[test['ObjectType'] == 3]
  del test['ObjectType']
  test['log_ctr'] = test['HistCTR'].apply(lambda x : -10 * round(log(x), 1))
  del test['HistCTR']
  print 'test.gl ingested, elapsed time: %s' % (datetime.now() - start)
  
  # join up test set
  print 'joining up test set...'
  ftest = test.join(si, how='left', on='SearchID')
  del ftest['SearchID']                          
  print 'join completed, elapsed time: %s' % (datetime.now() - start)
  
  # save test set
  path = os.path.join(avito2_io.PROCESSED, 'gl_test1.csv')
  print 'saving test features to %s' % path
  ftest.save(path, format='csv')
  print 'finished, elapsed time: %s' % (datetime.now() - start)

if __name__ == '__main__':
  features1()












