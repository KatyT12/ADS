# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
from .assess import *
from .access import *

def query_training_for_output_area(conn, geography_code, table='household_vehicle_data', pred_table='nssec_data', pred_var='L15', census_tags=['no_vehicle_ratio', 'one_vehicle_ratio']):
  query = f'''
    select *
    from code_count_table as cc
    join {table} as hv on hv.geography_code = cc.geography_code 
    join {pred_table} as ns on ns.geography_code = cc.geography_code 
    WHERE cc.geography_code = '{geography_code}';
  '''
  df = query_to_dataframe(conn, query)
  ret = extract_training_data(df, census_tags, pred_var)
  return ret

def find_nearest_output_areas(connection, latitude, longitude, number=1):
  query = f'''select geography_code, sqrt(pow(latitude - {latitude}, 2) + pow(longitude - {longitude}, 2)) as distance
          from nssec_data
          order by distance asc
          limit 1;'''
  
  return query_to_dataframe(connection, query)