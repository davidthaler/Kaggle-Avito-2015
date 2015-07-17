'''
The code in this file documents how the graphlab SFrame intermediate objects
were created. It also loads them. 

author: David Thaler
date: July 2015
'''
import graphlab as gl
import os
from avito2_io import DATA

GL_DATA = os.path.join(DATA, 'graphlab')


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
  ObjectType == 3 means that a row is a context ad, so we
  retain it.
  '''
  tr = load('train.gl')
  tr = tr[tr['ObjectType'] == 3]
  del tr['ObjectType']
  path = os.path.join(GL_DATA, 'train_context.gl')
  tr.save(path)


if __name__ == '__main__':
  context_ads()

















