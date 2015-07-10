import re
import pdb


KEY_PATTERN = r'(\d+):'

def jaccard_keys(search_params, ad_params):
  '''
  Computes Jaccard similarity between the keys of the Params for an ad
  and the SearchParams from SearchInfo.
  
  args:
    params1, params2 - the strings containing the Params, if any. 
        Expects an empty string if none.

  return:
    float, Jaccard similarity
  '''
  if len(search_params) == 0 or len(ad_params) == 0:
    return 0.0
  eps = 1e-6
  keys1 = set(re.findall(KEY_PATTERN, search_params))
  keys2 = set(re.findall(KEY_PATTERN, ad_params))
  #return round(len(keys1.intersection(keys2))/(eps + len(keys1.union(keys2))), 2)
  return len(keys1 - keys2) / (eps + len(keys1))