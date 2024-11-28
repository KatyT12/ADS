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

# To be used for both training and prediction of student proportions
def get_student_design_matrix(training):
  training['university_logged'] = np.log(training['university'].to_numpy()+1)
  training['dorm_logged'] = np.log(training['dormitory'].to_numpy()+1)
  training['apartments_logged'] = np.log(training['apartments'].to_numpy()+1)

  features = [training['university_logged'], training['no_vehicle_ratio'], training['apartments']]
  actual = training['training']
  X = np.column_stack(features)

  return sm.add_constant(X, has_constant='add'), actual

def query_training_for_output_area(conn, geography_codes, table='household_vehicle_data', pred_table='nssec_data', pred_var='L15', census_tags=['no_vehicle_ratio', 'one_vehicle_ratio']):
  codes = ', '.join([ '\'' + s + '\'' for s in geography_codes])
  query = f'''
    select *
    from code_count_table as cc
    join {table} as hv on hv.geography_code = cc.geography_code 
    join {pred_table} as ns on ns.geography_code = cc.geography_code 
    WHERE cc.geography_code in ({codes});
  '''
  df = query_to_dataframe(conn, query)
  ret = extract_training_data(df, census_tags, pred_var)
  return ret

def find_nearest_output_areas(connection, latitude, longitude, number=1):
  query = f'''select geography_code, sqrt(pow(latitude - {latitude}, 2) + pow(longitude - {longitude}, 2)) as distance
          from nssec_data
          order by distance asc
          limit {number};'''
  
  return query_to_dataframe(connection, query)