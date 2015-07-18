'''
The code in this file documents how the graphlab SFrame intermediate objects
were created. It also loads them. 

author: David Thaler
date: July 2015
'''
import graphlab as gl
import os
import avito2_io
from datetime import datetime
from random import random

GL_DATA = os.path.join(avito2_io.DATA, 'graphlab')


def load(infile):
  '''
  Reads a binary format SFrame from GL_DATA/
  
  args:
    infile - name of a graphlab binary to read from GL_DATA/
  
  return:
    the SFrame stored at GL_DATA/infile  
  '''
  path = os.path.join(GL_DATA, infile)
  return gl.load_sframe(path)
  
  
def write(infile, outfile, hints=int):
  '''
  Reads a data (.tsv) file from DATA/infile and writes out a binary at
  GL_DATA/outfile.
  
  args:
    infile - name of a .tsv file to read at DATA
    outfile - name of a graphlab binary to save at GL_DATA
    hints - type hints for columns, default is int
    
  return:
    None, but writes data into a binary SFrame
  '''
  inpath = os.path.join(GL_DATA, infile)
  print 'reading %s' % inpath
  data = gl.SFrame.read_csv(inpath, delimiter='\t', column_type_hints=hints)
  outpath = os.path.join(GL_DATA, outfile)
  print 'writing %s' % outpath
  data.save(outpath)
  
  
def write_all():
  '''
  This function executes (and records) the steps from loading the .tsv files
  into graphlab SFrames.
  '''
  write('UserInfo.tsv', 'user.gl')
  write('Category.tsv', 'category.gl')
  write('Location.tsv', 'location.gl')
  hints = {'SearchID'   : int,
           'AdID'       : int,
           'Position'   : int,
           'ObjectType' : int,
           'HistCTR'    : float,
           'IsClick'    : int}
  write('trainSearchStream.tsv', 'train.gl', hints=hints)
  del hints['IsClick']
  hints['ID'] = int
  write('testSearchStream.tsv', 'test.gl', hints=hints)
  hints = {'AdID'       : int,
           'LocationID' : int,
           'CategoryID' : int,
           'Params'     : dict,
           'Price'      : float,
           'Title'      : str,
           'IsContext'  : int}
  write('AdsInfo.tsv', 'ads.gl', hints=hints)
  hints = {'SearchID'       : int,
           'SearchDate'     : str,
           'IPID'           : int,
           'UserID'         : int,
           'IsUserLoggedOn' : int,
           'SearchQuery'    : str,
           'LocationID'     : int,
           'CategoryID'     : int,
           'SearchParams'   : dict}
  write('SearchInfo.tsv', 'search.gl', hints=hints)
  
  
def context_ads():
  '''
  This function records how context_ads.gl (at GL_DATA/) was created. 
  It requires that write_all was called earlier.
  '''
  ads = load('ads.gl')
  ctx = ads[ads['IsContext']]
  del ctx['IsContext']
  del ctx['LocationID']
  ctx['Price'] = ctx['Price'].fillna(0)
  path = os.path.join(GL_DATA, 'context_ads.gl')
  ctx.save(path)


def train_context():
  '''
  This function records how train_context.gl was created.
  ObjectType == 3 means that a row is a context ad, so we retain it.
  '''
  tr = load('train.gl')
  tr = tr[tr['ObjectType'] == 3]
  del tr['ObjectType']
  path = os.path.join(GL_DATA, 'train_context.gl')
  tr.save(path)


def test_context():
  '''
  This function records how test_context.gl was created.
  ObjectType == 3 means that a row is a context ad, so we retain it.
  '''
  test = load('test.gl')
  test = test[test['ObjectType'] == 3]
  del test['ObjectType']
  path = os.path.join(GL_DATA, 'test_context.gl')
  test.save(path)


def val_context():
  '''
  This function filters the rows of train_context.gl to just those rows 
  that are in the validation set(train_context() has to be run first).
  '''
  start = datetime.now()
  val_ids = avito2_io.get_artifact('full_val_set.pkl')
  tr = load('train_context.gl')
  idx = tr['SearchID'].apply(lambda id : id in val_ids)
  val = tr[idx]
  path = os.path.join(GL_DATA, 'val_context.gl')
  val.save(path)
  print 'elapsed time: %s' % (datetime.now() - start)
  
  
def search_val():
  '''
  This function filters the rows of search.gl (the SFrame containing
  SearchInfo.tsv) to just the rows used in the validation set.
  '''
  start = datetime.now()
  val_ids = avito2_io.get_artifact('full_val_set.pkl')
  si = load('search.gl')
  idx = si['SearchID'].apply(lambda x : x in val_ids)
  si_val = si[idx]
  path = os.path.join(GL_DATA, 'search_val.gl')
  si_val.save(path)
  print 'elapsed time: %s' % (datetime.now() - start)
  
  
def train_ds(p=0.05):
  '''
  Filters train_context.gl such that all of the positive rows are kept,
  but the negatives are selected with probability p. Also removes any 
  rows that are in the validation set.
  '''
  start = datetime.now()
  val_ids = avito2_io.get_artifact('full_val_set.pkl')
  tr1 = load('train_context.gl')
  idx1 = tr1['IsClick'].apply(lambda x : 1 if random() < p else x)
  tr2 = tr1[idx1]
  idx2 = tr2['SearchID'].apply(lambda x : x not in val_ids)
  tr_ds = tr2[idx2]
  path = os.path.join(GL_DATA, 'train_ds.gl')
  tr_ds.save(path)
  print 'elapsed time: %s' % (datetime.now() - start)


def search_ds():
  '''
  This filters search.gl to just the SearchIDs used in train_ds.gl.
  '''
  start = datetime.now()
  si = load('search.gl')
  tr_ds = load('train_ds.gl')
  ids = set(tr_ds['SearchID'])
  idx = si['SearchID'].apply(lambda x : x in ids)
  si_ds = si[idx]
  path = os.path.join(GL_DATA, 'search_ds.gl')
  si_ds.save(path)
  print 'elapsed time: %s' % (datetime.now() - start)  


def search_test():
  '''
  This filters search.gl to rows used in test.gl
  '''
  start = datetime.now()
  test = load('test.gl')
  si = load('search.gl')
  ids = set(test['SearchID'])
  idx = si['SearchID'].apply(lambda x : x in ids)
  si_test = si[idx]
  path = os.path.join(GL_DATA, 'search_test.gl')
  si_test.save(path)
  print 'elapsed time: %s' % (datetime.now() - start)  
  
  
if __name__ == '__main__':
  search_test()

















