'''
This script reads the AdsInfo.tsv file and creates a dict of the format:
{AdID: [CategoryID, Price, Title, Params]}
for each of the ~29k context ads. It is saved as ARTIFACTS/context_ads.pkl.

author: David Thaler
date: July 2015
'''

from datetime import datetime
import csv
import avito2_io

def parse_ads(maxlines=None):
  '''
  Scans the AdsInfo.tsv file and filters out all except the 28570 rows of
  contextual ads, and returns result as a dict.
  Format is {AdID: [CategoryID, Price, Title, Params]}.
  Price and *ID are ints; Params and Title are strings.
  
  args:
    maxlines - default None. The max # of lines to read out of AdsInfo.tsv.
        If None, read all lines.

  return:
    a dict with AdID of context ads as keys and a list of the meaningful
    fields in AdsInfo.tsv as values.
  '''
  out = {}
  with open(avito2_io.ADS_INFO) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for (k, line) in enumerate(reader):
      if k == maxlines:
        break
      if (int(line['IsContext']) == 1):
        if line['Price'] == '':
          line['Price'] = -1
        if line['CategoryID'] == '':
          line['CategoryID'] = -1
        values = [int(line['CategoryID']),
                  int(float(line['Price'])),
                  line['Title'],
                  line['Params']]
        out[int(line['AdID'])] = values
      if (k + 1) % 1000000 == 0:
        print 'read %d lines' % (k + 1)
  return out

  
if __name__=='__main__':
  start = datetime.now()
  print 'parsing AdsInfo.tsv'
  out = parse_ads() 
  print 'saving context ads to ARTIFACTS/'
  avito2_io.put_artifact(out, 'context_ads.pkl')
  print 'Finished, elapsed time: %s' % (datetime.now() - start)
  
  