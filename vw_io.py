import csv
import gzip
import argparse
import time
import string
import avito2_io
import os
from datetime import datetime
from math import log
import pdb

VWDATA = os.path.join(avito2_io.DATA, 'vwdata')

'''
THE PLAN:
  1 call join -> it1
  2 call process with it1 -> it2
  3 line = it2.next
  4 names = gen namespace(line, numerics)
  5 template = generate_template(line)
  6 call join again (reset)-> it3
  7 call process with it3 (reset)> it4
  8 call write_vw_output with outfile, it4, template, names
  9 repeat 6-8 with the other files (val set or test)
'''

def run():
  train  = os.path.join(VWDATA, 'tr_ds.csv.gz')
  search = os.path.join(VWDATA, 'si_ds.csv.gz')
  tr_out = os.path.join(VWDATA, 'train1.vw.gz')
  val = os.path.join(VWDATA, 'val_context.csv.gz')
  val_search = os.path.join(VWDATA, 'search_val.csv.gz')
  val_out = os.path.join(VWDATA, 'val1.vw.gz')
  test = os.path.join(VWDATA, 'test_context.csv.gz')
  test_search = os.path.join(VWDATA, 'search_test.csv.gz')
  test_out = os.path.join(VWDATA, 'test1.vw.gz')
  it1 = join(train, search)
  it2 = process(it1)
  line = it2.next()
  names = generate_namespaces(line)
  tmpl  = generate_template(line)
  it3 = join(train, search)
  lines = process(it3)
  print 'starting train'
  write_vw_output(tr_out, lines, tmpl, names)
  it4 = join(val, val_search)
  lines = process(it4)
  print 'starting validation set'
  write_vw_output(val_out, lines, tmpl, names)
  it5 = join(test, test_search)
  lines = process(it5)
  print 'starting test set'
  write_vw_output(test_out, lines, tmpl, names)
  

def join(tss, si, delimiter=','):
  '''
  NB: SearchID in tss and si are strings of int.
      The IDs in ctx, loc and cat are ints.
  '''
  ctx = avito2_io.get_artifact('context_ads_map.pkl')
  loc = avito2_io.get_artifact('location_map.pkl')
  cat = avito2_io.get_artifact('cat_map.pkl')
  with gzip.open(si) as f_si:
    with gzip.open(tss) as f_t:
      read_t  = csv.DictReader(f_t,  delimiter=delimiter)
      read_si = csv.DictReader(f_si, delimiter=delimiter)
      si_line = read_si.next()
      for (k, t_line) in enumerate(read_t):
        search_id = t_line['SearchID']
        while search_id != si_line['SearchID']:
          si_line = read_si.next()
          # Now the SearchID's match
        # NB: ad before si overwrites ad.CategoryID
        ad_id = int(t_line['AdID'])
        t_line.update(ctx[ad_id])
        t_line.update(si_line)
        loc_id = int(si_line['LocationID'])
        t_line.update(loc[loc_id])
        cat_id = int(si_line['CategoryID'])
        t_line.update(cat[cat_id])
        yield t_line


def process(in_lines):
  for line in in_lines:
    if 'ID' in line:
      line['label'] = -1
      line['tag'] = line['ID']
    else:
      line['label'] = line['IsClick']
      line['tag'] = line['IsClick']
      del line['IsClick']
    del line['SearchID']
    
    # this gets something running so we can end-to-end
    del line['SearchDate']
    del line['SearchParams']
    del line['Params']
    del line['Title']
    
    line['HistCTR'] = str(log((float(line['HistCTR']) + 1e-6)/ 0.006))
    if 'Price' not in line:
      line['Price'] = 0.0
    line['Price'] = str(log(1+ float(line['Price'])))
    yield line
    

def write_vw_output(outfile, in_lines, template, names):
  with gzip.open(outfile, 'w') as f_out:
    for line in in_lines:
      line.update(names)
      try:
        output = template.format(**line)
      except:
        pdb.set_trace()
      f_out.write(output)


# NB: these should probably only run once, per feature-design, to prevent
# the namespaces from being shuffled by non-deterministic key-ordering.

def generate_template(line_in, numerics=None):
  """
  Generates a template string for each line in the VW-formatted feature file.
  The format is: 
  
  <label> <tag>|<namespace_f1> f<f1> |<namespace_f2> f<f2>....
  |  <name_fn1>:<value fn1> | <name_fn2>:<value_fn2>...
  
  There is one namespace per non-numeric feature, with all such features 
  converted to strings (eg their feature value is 1). Numeric features get 
  a single namespace, and each feature has a name and value. This string <template> 
  can be used with a <line> from a csv.DictReader or similar as:
  
      outfile.write(<template>.format(**line)
  
  NB: line must have all features
  
  Params:
    line - one line of output like csv.DictReader output
    numerics - a space-separated string of field names of numeric values
            
  Returns:
    a template string that formats a line from a csv.DictReader into VW-format
  """
  line = line_in.copy()
  num_keys = None
  if numerics is not None:
    num_keys = set(numerics.split())
  line.pop('label')
  line.pop('tag')
  out = "{label} '{tag}"
  for key in line:
    if numerics is None or key not in num_keys:
      out += ' |{name_' + key + '} f{' + key + '}'
  if numerics is not None:
    out += ' |z'
    for key in num_keys.intersection(line):
      out += ' {name_' + key + '}:{' + key + '}'
  out += '\n'
  return out


def generate_namespaces(line):
  """
  Generates one VW namespace or name for every column in an input csv file.
  The names will be unique single letters, since VW only looks at the
  first character of namespaces.
  
  NB: line must have all features
  
  Params:
    line - one line of output like csv.DictReader output
            
  Returns:
    a dictionary of the form {namespace name: unique letter}
  """
  line = line.copy()
  line.pop('label')
  line.pop('tag')
  keys = ['name_' + name for name in line.keys()]
  out = dict(zip(keys, string.letters))
  return out

if __name__=='__main__':
  start = datetime.now()
  run()
  print 'elapsed time: %s' % (datetime.now() - start)




