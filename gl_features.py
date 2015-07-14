import graphlab as gl
import sframes
import avito2_io
import os
from math import log, ceil
from datetime import datetime


def get_context_ads():
  '''
  Loads the ads.gl file (all of AdsInfo.tsv) and extracts the context ads.
  '''
  ads = sframes.load('ads.gl')
  ctx = ads[ads['IsContext']]
  del ctx['IsContext']
  del ctx['LocationID']
  return ctx

def features2():
  '''
  This function implements and records the construction of the second set of 
  graphlab sframe features. These include all of the integer-valued raw and
  lightly processed features from train/test, SearchInfo, Category, Location,
  AdsInfo and UserInfo. Only contextual ads are considered.
  
  NB: This leaves SearchID in the output(to allow for validation).
      Run script must delete SearchID.
  '''
  start = datetime.now()
  print 'loading context ads'
  ctx = get_context_ads()
  ctx['LogPrice'] = ctx['Price'].apply(lambda x : round(log(x+1), 1))
  ctx['ParamLen'] = ctx['Params'].apply(lambda d : len(d)).fillna(0)
  ctx['TitleLen'] = ctx['Title'].apply(lambda s : len(s)).fillna(0)
  del ctx['Price']
  del ctx['Title']
  del ctx['Params']
  print 'loading users'
  users = sframes.load('user.gl')
  print 'loading category and location'
  ctg = sframes.load('category.gl')
  # Admins said this field could be deleted.
  del ctg['SubcategoryID']
  loc = sframes.load('location.gl')
  print 'small objects loaded, elapsed time: %s' % (datetime.now() - start)
  
  print 'ingesting train.gl'
  tr = sframes.load('train.gl')
  tr = tr[tr['ObjectType'] == 3]
  del tr['ObjectType']
  tr['log_ctr'] = tr['HistCTR'].apply(lambda x : -10 * round(log(x), 1))
  del tr['HistCTR']
  print 'train.gl ingested, elapsed time: %s' % (datetime.now() - start) 
  
  # SearchDate: I can't decide what to do with it, so I'm leaving it in, 
  # as-is. The run script will have to remove it. This allows sorting by
  # date without doing this huge join again.
  
  print 'ingesting search.gl'
  si = sframes.load('search.gl')  
  si['SQexists'] = si['SearchQuery'].apply(lambda x : len(x) > 0)
  del si['SearchQuery']
  si['SPexists'] = (si['SearchParams'].apply(lambda d : int(d is not None))
                                      .fillna(0))
  del si['SearchParams']
  print 'search.gl ingested, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining user.gl into search.gl'
  si = si.join(users, how='left', on='UserID')
  print 'user.gl joined in, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining location.gl to search.gl'
  si = si.join(loc, how='left', on='LocationID')
  print 'location.gl joined in, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining category.gl to search.gl'
  si = si.join(ctg, how='left', on='CategoryID')
  print 'category.gl joined in, elapsed time: %s' % (datetime.now() - start)
  
  # join category into context ads and rename to avoid name clash
  print 'joining category into ads'
  ctx = ctx.join(ctg, how='left', on='CategoryID')
  ctx.rename({'CategoryID':'AdCat'})
  print 'category.gl joined into ads, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining context ads into train'
  tr = tr.join(ctx, how='left', on='AdID')
  print 'context ads joined into train, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining up training set (search and train)...'
  tr = tr.join(si, how='left', on='SearchID')
  print 'join completed, elapsed time: %s' % (datetime.now() - start)
  
  print 'sorting train by SearchDate, SearchID, AdID
  tr = tr.sort(['SearchDate', 'SearchID', 'AdID'])
  
  path = os.path.join(avito2_io.PROCESSED, 'gl_train2.csv')
  print 'saving training features to %s' % path
  tr.save(path, format='csv')
  print 'training features saved, elapsed time: %s' % (datetime.now() - start)
  
  # test
  
  print 'ingesting test.gl'
  test = sframes.load('test.gl')
  test = test[test['ObjectType'] == 3]
  del test['ObjectType']
  test['log_ctr'] = test['HistCTR'].apply(lambda x : -10 * round(log(x), 1))
  del test['HistCTR']
  print 'test.gl ingested, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining context ads into test'
  test = test.join(ctx, how='left', on='AdID')
  print 'context ads joined into test, elapsed time: %s' % (datetime.now() - start)
  
  print 'joining up test set...'
  ftest = test.join(si, how='left', on='SearchID')
  del ftest['SearchID']                          
  print 'join completed, elapsed time: %s' % (datetime.now() - start)
  
  print 'sorting test...'
  ftest = ftest.sort('ID')
  
  path = os.path.join(avito2_io.PROCESSED, 'gl_test2.csv')
  print 'saving test features to %s' % path
  ftest.save(path, format='csv')
  print 'finished, elapsed time: %s' % (datetime.now() - start)


if __name__ == '__main__':
  features2()


# Probably dead code


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











