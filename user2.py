'''
This script records the steps for creating the user2.pkl file in ARTIFACTS.
It produces a dict like: {SearchID: [UserID, SearchDate, CategoryID]}
Fields *ID are ints and the date is a datetime.datetime.

author: David Thaler
date: July 2015
'''
from avito2_io import *
etl = [lambda line : int(line['UserID']),
       lambda line : convert_date(line['SearchDate']),
       lambda line : int(line['CategoryID'])]
out = sample_search_info_by_user(100, etl=etl)
put_artifact(out, 'user2.pkl')