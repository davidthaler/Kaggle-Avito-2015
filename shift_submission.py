from math import log, exp
import numpy as np
import pandas as pd
import avito2_io
import os


def shift(submit_id, new_mean, is_gzip=True):
  if is_gzip:
    sub_file_name = 'submission%s.csv.gz' % str(submit_id)
    submit_path = os.path.join(avito2_io.SUBMIT, sub_file_name)
    sub = pd.read_csv(submit_path, compression='gzip')
  else:
    sub_file_name = 'submission%s.csv' % str(submit_id)
    submit_path = os.path.join(avito2_io.SUBMIT, sub_file_name)
    sub = pd.read_csv(submit_path)
  old_mean = sub.IsClick.mean()
  offset = log(new_mean/(1.0 - new_mean)) - log(old_mean/(1.0 - old_mean))
  p = sub.IsClick.values
  dv = np.log(p/(1.0 - p))
  dv += offset
  p = 1.0/(1.0 + np.exp(-dv))
  sub['IsClick'] = p
  return sub
  
def shift_df(sub, new_mean):
  old_mean = sub.IsClick.mean()
  offset = log(new_mean/(1.0 - new_mean)) - log(old_mean/(1.0 - old_mean))
  p = sub.IsClick.values
  dv = np.log(p/(1.0 - p))
  dv += offset
  p = 1.0/(1.0 + np.exp(-dv))
  out =  sub.copy()
  out['IsClick'] = p
  return out