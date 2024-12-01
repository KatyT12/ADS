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

# Fit a binomial model
# design_func is the function which retrieves the design matrix
def fit_binomial_model(connection, number, design_func):
  print('Collecting training data')
  df = fynesse.assess.query_random_set(connection, number)
  training = fynesse.assess.extract_training_data(df)
  X, actual = design_func(training)
  print('Fitting model')
  model = sm.GLM(actual, X, family = sm.families.Binomial())
  results = model.fit()
  return (results, X, training)



# Extract data from the database into a useful table that can be used for training and finding correlations
def extract_training_data(df, census_tags=['no_vehicle_ratio', 'one_vehicle_ratio', 'two_vehicle_ratio'], pred_var='L15'):
  df = df.loc[:,~df.columns.duplicated()].copy()
  codes = df['geography_code'].drop_duplicates()
  osm_tags = df['tag'].drop_duplicates()
  result = df.pivot(index='geography_code', columns='tag', values='count').reset_index()
  for t in osm_tags:
    result.loc[result[t].isnull(), t] = 0

  if pred_var == 'L15':
    result = result.merge(df[['geography_code', *census_tags, 'L15', 'total_over_16']].drop_duplicates(), left_on='geography_code', right_on='geography_code')
    result['training'] = result['L15']/result['total_over_16']
  else:
    result = result.merge(df[['geography_code', *census_tags, pred_var]].drop_duplicates(), left_on='geography_code', right_on='geography_code')
    result['training'] = result[pred_var]
  return result