'''
This file contains input output code that runs against the large .tsv files.

author: David Thaler
date: July 2015
'''
import graphlab as gl
import numpy as np
import pandas as pd
import sframes
import avito2_io
import pdb


def basic_join(tss, si):
  '''
  A generator that performs a rolling join over train_context.gl and search.gl, 
  which are graphlab binary files storing trainSearchStream.tsv 
  (after removal of ads other than context ads) and SearchInfo.tsv, respectively.
  SFrame context_ads.gl, which has the contextual ads from AdsInfo.tsv, is
  also joined in.
  
  args:
    tss - an SFrame with data from trainSearchStream or testSearchStream, 
        including samples or validation sets
    si - an SFrame with data from SearchInfo. Must have all of the SearchIDs
        in tss, but it can be a sample
        
  generates:
    a dict that combines all of the fields from tss, si and ads for a row
  '''
  ctx = sframes.load('context_ads.gl')
  ctx = sframe_to_dict('AdID', ctx)
  si_it = iter(si)
  si_line = si_it.next()
  for tss_line in tss:
    search_id = tss_line['SearchID']
    ad_id = tss_line['AdID']
    while search_id != si_line['SearchID']:
      si_line = si_it.next()
    # Now the SearchIDs match
    tss_line.update(ctx[ad_id])
    # SearchInfo.CategoryID overwrites AdInfo.CategoryID in this line
    tss_line.update(si_line)
    yield tss_line
  
  
def sframe_to_dict(key_field, sframe):
  '''
  Converts the rows of a graphlab SFrame to a dict of dicts of the form:
    {id : {field_name: field_value, ...}}
  
  args:
    id - the name of the field to use as the dictionary key
    sframe - the Graphlab SFrame to get the dict entries out of
    
  return:
    a dict like {id:{field_name:field_value, ...}}
  '''
  d = {}
  for row in sframe:
    id = row.pop(key_field)
    d[id] = row
  return d




