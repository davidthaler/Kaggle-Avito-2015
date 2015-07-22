import csv
import gzip
import argparse
import time
import pdb


BASE = '/Users/davidthaler/Documents/Kaggle/avazu/'
DATA = BASE + 'data/'
VWDATA = DATA + 'vwdata/'
TMP = BASE + 'tmp/'
SUBMISSION = BASE + 'submissions/submission_%d.csv'
SITE_ID_NULL = '85f751fd'


NAMESPACES = {
 'name_C1': 'O',
 'name_C14': 'L',
 'name_C15': 'K',
 'name_C16': 'M',
 'name_C17': 'I',
 'name_C18': 'F',
 'name_C19': 'E',
 'name_C20': 'V',
 'name_C21': 'A',
 'name_app_category': 'P',
 'name_app_domain': 'R',
 'name_app_id': 'D',
 'name_banner_pos': 'T',
 'name_click': 'J',
 'name_device_conn_type': 'N',
 'name_device_id': 'U',
 'name_device_ip': 'B',
 'name_device_model': 'X',
 'name_device_type': 'G',
 'name_hour': 'W',
 'name_id': 'H',
 'name_site_category': 'Q',
 'name_site_domain': 'S',
 'name_site_id': 'C'}


TEMPLATE  = "{click} {tag}|{name_C1} f{C1} |{name_C14} f{C14} |{name_C15} f{C15} "
TEMPLATE += "|{name_C16} f{C16} |{name_C17} f{C17} |{name_C18} f{C18} "
TEMPLATE += "|{name_C19} f{C19} |{name_C20} f{C20} |{name_C21} f{C21} "
TEMPLATE += "|{name_app_category} f{app_category} |{name_app_domain} f{app_domain} "
TEMPLATE += "|{name_app_id} f{app_id} |{name_banner_pos} f{banner_pos} "
TEMPLATE += "|{name_device_conn_type} f{device_conn_type} "
TEMPLATE += "|{name_device_id} f{device_id} |{name_device_ip} f{device_ip} "
TEMPLATE += "|{name_device_model} f{device_model} |{name_device_type} f{device_type} "
TEMPLATE += "|{name_hour} f{hour} |{name_site_category} f{site_category} "
TEMPLATE += "|{name_site_domain} f{site_domain} |{name_site_id} f{site_id}\n"

# modified for numeric data on end
TEMPLATE = TEMPLATE[:-1] + " "
TEMPLATE += "|Z device_app_ct:{dip_app_ct} device_app_uniq:{dip_app_u} "
TEMPLATE +=    "device_web_ct:{dip_web_ct} device_web_uniq:{dip_web_u} "
TEMPLATE += "app_vol:{app_traffic} web_vol:{web_traffic} all_vol:{all_traffic} "
TEMPLATE += "webct:{webct} appct:{appct} ratio:{ratio}\n"

def get_csv(infile, maxlines, is_test):
  if infile.endswith('.gz'):
    op_fn = gzip.open
  else:
    op_fn = open
  with op_fn(infile) as f_in:
    reader = csv.DictReader(f_in)
    for (k, line) in enumerate(reader):
      if (maxlines is not None) and (k >= maxlines):
        break
      if is_test:
        line['click'] = '-1'
        line['tag'] = line['id']
      else:
        line['click'] = str(2 * int(line['click']) - 1)
        line['tag'] = line['click']
      # This breaks separation of the last two days validation set
      #line['hour'] = line['hour'][6:]
      line.update(NAMESPACES)
      yield line


def munge_vw(infile, outfile, maxlines, is_test):
  f_out = open(outfile, 'w')
  reader = get_csv(infile, maxlines, is_test)
  for line in reader:
    output = TEMPLATE.format(**line)
    f_out.write(output)
  f_out.close()


def munge_split(infile, webpath, mobilepath, is_test):
  f_web = open(webpath, 'w')
  f_mobile = open(mobilepath, 'w')
  reader = get_csv(infile, maxlines=None, is_test=is_test)
  for line in reader:
      output = TEMPLATE.format(**line)
      if line['site_id'] == SITE_ID_NULL:
        f_mobile.write(output)
      else:
        f_web.write(output)
  f_web.close()
  f_mobile.close()
  


def split4(infile, vwpath):
  tr_web = open(vwpath + 'train.web.vw', 'w')
  tr_mobile = open(vwpath + 'train.mobile.vw', 'w')
  val_web = open(vwpath + 'val.web.vw', 'w')
  val_mobile = open(vwpath + 'val.mobile.vw', 'w')
  reader = get_csv(infile, maxlines=None, is_test=False)
  for line in reader:
      output = TEMPLATE.format(**line)
      if line['hour'].startswith('141029') or line['hour'].startswith('141030'):
        f_mobile = val_mobile
        f_web = val_web
      else:
        f_mobile = tr_mobile
        f_web = tr_web
      if line['site_id'] == SITE_ID_NULL:
        f_mobile.write(output)
      else:
        f_web.write(output)
  tr_web.close()
  val_web.close()
  tr_mobile.close()
  val_mobile.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=\
    'Parses the csv files for the Kaggle Avazu competition to vw format')
  start = time.time()
  parser.add_argument('infile', metavar='IN', type=str,
                      help='The gzip file with the input data in csv format')
  parser.add_argument('outfile', metavar='OUT', type=str,
                      help='The location to write out the vw-formatted output')
  parser.add_argument('-m', '--maxlines', type=int,
                      help='Only read this many lines of the input')
  parser.add_argument('-t', '--test', dest='test', action='store_const',
                      const=True, default=False,
                      help='Treat this file as a test file, adding 0 labels to it.')
  args = parser.parse_args()
  munge_vw(args.infile, args.outfile, args.maxlines, args.test)
  et = time.time() - start
  print 'runtime: %d sec.' % et






