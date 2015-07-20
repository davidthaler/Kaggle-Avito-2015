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


def basic_join(tss, si, user):
  '''
  A generator that performs a rolling join over Graphlab SFrames tss, which
  stores data from train/testSearchStream.tsv and si, which is from 
  SearchInfo.tsv. SFrame context_ads.gl, which has the contextual ads from 
  AdsInfo.tsv, is loaded and joined in. UserInfo.tsv is joined in from loading
  the artifact user_dict.pkl from artifacts/.
  
  args:
    tss - an SFrame with data from trainSearchStream or testSearchStream, 
        including samples or validation sets
    si - an SFrame with data from SearchInfo. Must have all of the SearchIDs
        in tss, but it can be a sample
    user -  dict or None. A dict from UserID to a dict of features for 
        that user. Caller should construct this if used.
        
  generates:
    a dict that combines all of the fields from tss, si and ads for a row
  '''
  ctx = sframes.load('context_ads.gl')
  ctx = sframes.sframe_to_dict('AdID', ctx)
  si_it = iter(si)
  si_line = si_it.next()
  for tss_line in tss:
    search_id = tss_line['SearchID']
    ad_id = tss_line['AdID']
    user_id = si_line['UserID']
    while search_id != si_line['SearchID']:
      si_line = si_it.next()
    # Now the SearchIDs match
    tss_line.update(ctx[ad_id])
    # SearchInfo.CategoryID overwrites AdInfo.CategoryID in this line
    tss_line.update(si_line)
    if user is not None and user_id in user:
      tss_line.update(user[user_id])
    yield tss_line
  




