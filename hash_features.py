'''
Standard Python only (no numpy/scipy, pandas or sklearn) feature hasher based 
on Sam Hocevar's Avazu benchmark code. Avoiding non-standard libraries means 
that we can use Pypy for a large speedup.

author: David Thaler
date: July 2015
'''
def hash_features(x, D):
  '''
  Performs a sparse one-hot encoding of a dictionary of input features such
  as from csv.DictReader.
  
  args:
    x - a dictionary of features line the output of csv.DictReader.
    D - the maximum index of any feature, or dimension of feature space
    
  returns:
    a list of indices of features that are non-zero (one)
  '''
  return [abs(hash(str(key) + '_' + str(x[key]))) % D for key in x]
