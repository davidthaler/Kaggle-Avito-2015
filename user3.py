'''
This script records the steps for creating the user3.pkl file in ARTIFACTS.
It collects a 5% sample by value of UserIDs, and produces a dict like: 
{SearchID: [UserID, 
            SearchDate, 
            CategoryID, 
            IsLoggedOn, 
            SearchQuery exists,
            SearchParams exists]}
Fields *ID are ints and the date is a datetime.datetime. 
The last 3 fields are 0/1 ints.

author: David Thaler
date: July 2015
'''
from avito2_io import *
from datetime import datetime

start = datetime.now()
etl = [lambda line : int(line['UserID']),
       lambda line : convert_date(line['SearchDate']),
       lambda line : int(line['CategoryID']),
       lambda line : int(line['IsUserLoggedOn']),
       lambda line : int(len(line['SearchQuery']) > 0),
       lambda line : int(len(line['SearchParams']) > 0) ]
out = sample_search_info_by_user(20, etl=etl)
put_artifact(out, 'user3.pkl')
print 'elapsed time: %s' % (datetime.now() - start)