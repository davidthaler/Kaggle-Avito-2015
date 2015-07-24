import subprocess
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import log_loss
import avito2_io
import pdb

BASE = avito2_io.BASE
DATA = avito2_io.DATA
VWDATA = BASE + 'vwdata/'
TMP = BASE + 'tmp/'
SUBMISSION = BASE + '/submissions/submission_%d.csv'
SAMPLE = DATA + '/sampleSubmission.csv'


def sigmoid(x):
  return 1./(1. + np.exp(-x))


def inv_sigmoid(x):
  return np.log(x/(1.-x))
  
  
def vw_train(data_file, 
             l2=None, 
             l1=None, 
             keep=None,
             ignore=None,
             quadratic=None,
             passes=None, 
             model_file=None,
             learn_rate=0.1,
             holdout=False,
             other=None):
  """
  Function programmatically calls VW for training.
  Optionally writes out the model file containing the learned weights.
  
  Params:
    data_file - the training data in VW's input format
    l2 - (optional) the L2 regularization parameter 
    l1 - (optional) the L1 regularization parameter
       NOTE: In VW, this is per row, so it should be small ~ 1e-7
    keep - (optional) string with the first letters of namespaces to use,
           others are ignored. Default is use all namespaces.
    ignore - (optional) string with the first letters of the namespaces to ignore.
             VW uses all of the others. Default is ignore none/use all.
       NOTE: At most one of keep and ignore can be specified.
    quadratic - string with first letters of all namespaces that should be crossed
              to make quadratic terms. Uses all pairs of these.
    passes - (optional) the number of passes to use. Default is 1.
    model_file - (optional) A file to write the final learned models out to.
              If not specified, training is run, and there is output,
              but no model is saved.
    learn_rate - default 0.5
    holdout - default True. Use VW defaults for holdout. If False, no holdout.
    other - A list of strings to pass to VW as command line arguments
    
  Returns:
    nothing, but writes out the final regressor at <model_file>
  """
  cmdline = ['vw',
             '-d', data_file,
             '-b', '26',
             '-l', str(learn_rate),
             '--loss_function', 'logistic',
             '--progress', '1000000']
  if l2 is not None:
    cmdline.extend(['--l2', str(l2)])
  if l1 is not None:
    cmdline.extend(['--l1', str(l1)])
  if passes is not None:
    cmdline.extend(['--passes', str(passes), '-c'])
  if model_file is not None:
    cmdline.extend(['-f', model_file])
  if keep is not None and ignore is not None:
    raise ValueError('At most one of --keep and --ignore can be specified.')
  if keep is not None:
    arg = '--keep'
    names = keep
  if ignore is not None:
    arg = '--ignore'
    names = ignore
  if keep is not None or ignore is not None:
    for n in names:
      cmdline.extend([arg, n])
  if quadratic is not None:
    for (a,b) in itertools.combinations(quadratic, 2):
      cmdline.extend(['-q', a+b])
  if not holdout:
    cmdline.append('--holdout_off')
  if other is not None:
    cmdline.extend(other)
  subprocess.call(cmdline)


def vw_predict(model_file, test_data, outfile=None):
  """
  Function programmatically calls VW. 
  Optionally saves the output in VW's output format:
  <prediction> <tag>
  where <prediction> will be of the log-odds scale.
  
  Params:
    model_file - the file resulting from a VW training run
    test_data - the data to predict on in VW input format
    outfile - Optional. VW will write results here.
  """
  cmdline = ['vw', 
             '-t', test_data,
             '-i', model_file,
             '--loss_function', 'logistic',
             '--progress', '1000000']
  if outfile is not None:
    cmdline.extend(['-p', outfile])
  subprocess.call(cmdline)
  
    
def score_val(infile, offset=0):
  """
  Takes a file with the format <prediction> <label> where the label
  is -1/1 and the predictions are in log-odds form and outputs the 
  log loss for those predictions and labels. The format is
  what VW outputs if the labels are the tag for the training set.
  
  Params:
    infile - path to the prediction file
    offset - an amount added to the log-odds of each prediction
             before computing probabilities
             
  Returns:
    the log loss for the predictions and labels in <infile>
  """
  data = read_vw_results(infile, offset)
  # For validation, the tag contains labels, not id's
  labels = (data['ID'] == 1).astype(float)
  return log_loss(labels.values, data.IsClick.values)
  
  
def write_vw_submission(submit_num, infile1, infile2=None, offset1=0, offset2=0):
  """
  Takes a file with the format <prediction> <ID> where the prediction
  is in log-odds form. Writes out a file suitable for submitting.
  
  Params:
    submit_num - output is submission_<submit_num>.csv
    infile1 - path to the file with test set predictions
    offset1 - an amount added to the log-odds of each prediction
             from infile1 before computing probabilities
    infile2 - path to a file with more test set predictions
    offset2 - an offset to apply to log-odds values in infile2
  """
  out = read_vw_results(infile1, offset1)
  submit_file = SUBMISSION % submit_num
  out[['ID','IsClick']].to_csv(submit_file, index=False)
  if infile2 is not None:
    out = read_vw_results(infile2, offset2)
    out[['ID','IsClick']].to_csv(submit_file, index=False, header=None, mode='a')
  print 'wrote ' + submit_file


def read_vw_results(infile, offset):
  """
  Factors out common code for accessing data in WV output format,
  which is <prediction> <tag>. Tag can be an example ID or a label.

  Params:
    infile - path to the file with test set predictions
    offset - an amount added to the log-odds of each prediction
             from infile1 before computing probabilities
  """
  data = pd.read_csv(infile, delimiter=' ', header=None)
  pred = data[0]
  pred += offset
  probs = sigmoid(pred)
  out = pd.DataFrame({'ID':data[1],'IsClick':probs})
  return out
  
  
def combine(submissions, weights=None, logspace=True):
  """
  Params:
    submissions - a list of pandas data frames containing submissions.
        The ID's need not be ordered concordantly.
    w - Sequence or array of weights for averaging the submissions. 
        Optional, default uniform. Weights will be normalized to sum to one. 
        Converted to np.array of dtype float.
    logspace - default True. Take the average on the log-odds scale.
        
  Returns:
    a pandas data frame that can be submitted in which the IsClick field
    is the weighted average of the IsClick field in <submissions>.
  """
  
  if weights is None:
    weights = np.ones(len(submissions))
  else:
    weights = np.array(weights, dtype=float)
  weights = weights/np.sum(weights)
  ss = pd.read_csv(SAMPLE)
  ss.rename(columns={'IsClick':'total'}, inplace=True)
  ss.total = 0.0
  for (w, sub) in zip(weights, submissions):
    ss = ss.merge(sub, on='ID')
    if logspace:
      ss.total = ss.total + w * ss.IsClick.apply(inv_sigmoid)
    else:
      ss.total = ss.total + w * ss.IsClick
    del ss['IsClick']
  if logspace:
    ss.total = ss.total.apply(sigmoid)
  ss.rename(columns={'total':'IsClick'}, inplace=True)
  return ss





  









