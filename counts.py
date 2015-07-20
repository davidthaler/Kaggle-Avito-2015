'''
The code in this file constructs occurrence counts from the .tsv files.
It is in pure Python so that we can use pypy.

author: David Thaler
date: July 2015
'''
import avito2_io
import gzip
import cPickle
import csv
import os
import argparse
from datetime import datetime
import pdb

def user_counts(min_ct):
  '''
  Collect counts of the UserID in SearchInfo, PhoneStream and VisitStream.
  
  args:
    min_ct - None or int. If not None, UserID must occur at least min_ct 
        times in SearchInfo.tsv or else it will be dropped.
        
  return:
     a dict of {int: dict} mapping UserID to statistics of that user
  '''
  out = {}
  with open(avito2_io.SEARCH_INFO) as f_si:
    reader = csv.DictReader(f_si, delimiter='\t')
    for (k, line) in enumerate(reader):
      user_id = int(line['UserID'])
      value = out.get(user_id, {'si_ct':0})
      value['si_ct'] += 1
      out[user_id] = value
      if (k + 1) % 1000000 == 0:
        print 'read %d lines from SearchInfo.tsv' % (k + 1)
  
  if min_ct is not None:
    out = {k:out[k] for k in out if out[k]['si_ct'] >= min_ct}
    
  with gzip.open(avito2_io.PHONE) as f_ph:
    reader = csv.DictReader(f_ph, delimiter='\t')
    for (k, line) in enumerate(reader):
      user_id = int(line['UserID'])
      if user_id in out:
        ph_ct = out[user_id].setdefault('ph_ct', 0)
        out[user_id]['ph_ct'] = ph_ct + 1
      if (k + 1) % 1000000 == 0:
        print 'read %d lines from PhoneRequestsStream.tsv.gz' % (k + 1)
  for k in out:
    out[k].setdefault('ph_ct', 0)
  
  with gzip.open(avito2_io.VISIT) as f_vis:
    reader = csv.DictReader(f_vis, delimiter='\t')
    for (k, line) in enumerate(reader):
      user_id = int(line['UserID'])
      if user_id in out:
        vis_ct = out[user_id].setdefault('vis_ct', 0)
        out[user_id]['vis_ct'] = vis_ct + 1
      if (k + 1) % 1000000 == 0:
        print 'read %d lines from VisitsStream.tsv.gz' % (k + 1)
  for k in out:
    out[k].setdefault('vis_ct', 0)

  return out
      
if __name__ == '__main__':
  start = datetime.now()
  print 'running at: ' + str(start)
  parser = argparse.ArgumentParser(description = 
                   'Collects counts of UserID from several data files.')
  parser.add_argument('--min_ct', type=int, default=None)
  parser.add_argument('--max_lines', type=int, default=None)
  args = parser.parse_args()
  user_counts = user_counts(args.min_ct)
  avito2_io.put_artifact(user_counts, 'user_counts.pkl')
  print 'elapsed time: %s' % (datetime.now() - start)







  