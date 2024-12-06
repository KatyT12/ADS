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
import statsmodels.api as sm

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

def query_for_location(connection, latitude, longitude, number, table):
  oa = find_nearest_output_areas(connection, latitude, longitude, number=number)
  codes = ', '.join([ '\'' + s + '\'' for s in oa['geography_code']])
  query = f'''
    select * from {table}
    where geography_code in ({codes});
  '''
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



# Plot the area and points of interest onto a graph
#   :param connection - The ongoing SQL connection
#   :param connection
#   :param ax: Axis to plot on
def get_avg_price_lad(connection, df, types):
  prices = fynesse.assess.get_avg_price(connection)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()
  merged = df.merge(prices, left_on='geography_code', right_on='lad23', how='left')
  
  med = merged['avg(price)'].median()
  return merged.fillna(med)

# For specific codes, or optionally for a random set of geography codes, which may be helpful later
def get_avg_price_oa(connection, df, types, codes=[], random_number=None, label ='geography_code'):
  query = ''
  if random_number is None:
    codes_string = ', '.join(["'" + s + "'" for s in codes])
    query = f'''
      with select
      select avg(price), stddev(price), oa23, property_type from pp_data_oa_joined group by lad23, property_type where oa in ({codes_string});
    '''
  else:
    query = f'''
    with codes as (select geography_code from nssec_data order by rand () limit {random_number})
    select avg(price), stddev(price), oa21, property_type from pp_data_oa_joined where oa21 in (select * from codes) group by oa21, property_type;
    '''
  prices = fynesse.assess.query_to_dataframe(connection, query)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()

  merged = df.merge(prices, left_on=label, right_on='oa21', how='left')
  med = merged['avg(price)'].median()
  return merged.fillna(med)








def augment_training(training, nimby_df,cols=['rag', 'avg_rag_flats', 'avg_rag_housing', 'avg_rag_estate']):
  merged = training.merge(nimby_df[['LAD23CD',*cols]], left_on='geography_code', right_on='LAD23CD')  
  return merged[merged['geography_code'].str.contains('E')]


# Retrieve the average price for each postcode, get the median
#   :param connection - The ongoing SQL connection
#   :param connection
#   :param ax: Axis to plot on
def get_avg_price_lad(connection, df, types):
  prices = fynesse.assess.get_avg_price(connection)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()
  merged = df.merge(prices, left_on='geography_code', right_on='lad23', how='left')
  
  med = merged['avg(price)'].median()
  return merged.fillna(med)

# For specific codes, or optionally for a random set of geography codes, which may be helpful later
def get_avg_price_oa(connection, df, types, codes=[], random_number=None, label ='geography_code'):
  query = ''
  if random_number is None:
    codes_string = ', '.join(["'" + s + "'" for s in codes])
    query = f'''
      with select
      select avg(price), stddev(price), oa23, property_type from pp_data_oa_joined group by lad23, property_type where oa in ({codes_string});
    '''
  else:
    query = f'''
    with codes as (select geography_code from nssec_data order by rand () limit {random_number})
    select avg(price), stddev(price), oa21, property_type from pp_data_oa_joined where oa21 in (select * from codes) group by oa21, property_type;
    '''
  prices = fynesse.assess.query_to_dataframe(connection, query)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()

  merged = df.merge(prices, left_on=label, right_on='oa21', how='left')
  med = merged['avg(price)'].median()
  return merged.fillna(med)


# Augment training data with the response variable, for convenience
def augment_training(training, nimby_df,cols=['rag', 'avg_rag_flats', 'avg_rag_housing', 'avg_rag_estate']):
  merged = training.merge(nimby_df[['LAD23CD',*cols]], left_on='geography_code', right_on='LAD23CD')
  return merged[merged['geography_code'].str.contains('E')]


def fit_model_OLS(connection, training, actual, t, design_func, augmented=None, ridge=None):
  if augmented is None:
    train = augment_training(training, actual)
    augmented = get_avg_price_lad(connection, train, ['T'])


  design = design_func(augmented, augmented)
  
  model = sm.OLS(augmented[t], design)
  

  if ridge is not None:
    fitted_model = model.fit_regularized(alpha=ridge, L1_wt=0)
    return fitted_model
  else:
    fitted_model = model.fit()
    return fitted_model



# Train and then predict on training data, plot correlation
#   :param connection - The ongoing SQL connection
#   :param training - training data which is adequate for design
#   :param actual - The actual values trying to predict
#   :param design_func - the function for generating a design matrix
#   :param t - specific response variable of interest
def predict_model_against_training(connection, training, nimby_df, model, design_func, t, ax = None, model_name='', augmented=None):

  # Augment with actual data and house prices
  if augmented is None:
    train = augment_training(training, nimby_df)
    augmented = get_avg_price_lad(connection, train, ['T'])

  # Retrieve predictions
  predicted = model.predict(design_func(augmented, augmented))
  actual = augmented[t].to_numpy()

  # Plot correlation
  if ax is None:
    fig, ax = plt.subplots()

  corr = np.corrcoef(predicted, actual)
  sns.regplot(x=actual, y=predicted, ax=ax, label='Best fit')
  #ax.scatter(predicted, actual)
  ax.set_title(f'Correlation for {t} ({round(corr[0][1], 3)})')
  ax.set_xlabel(f'Actual {t}')
  ax.set_xlabel(f'Predicted {t}')


# Augment LAD census data with the average price. Can affort a full group by SQL query
def get_avg_price_lad(connection, df, types):
  # SQL join
  prices = get_avg_price(connection)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  # Merge, aggregate by property type. Slightly dodgy aggre
  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()
  merged = df.merge(prices, left_on='geography_code', right_on='lad23', how='left')

  med = merged['avg(price)'].median()
  return merged.fillna(med)



# For specific codes, or optionally for a random set of geography codes, which may be helpful later
def augment_avg_price_oa(connection, df, types, codes=[], random_number=None, label ='geography_code'):
  query = ''
  if random_number is None:
    codes_string = ', '.join(["'" + s + "'" for s in codes])
    query = f'''
      with select
      select avg(price), stddev(price), oa23, property_type from pp_data_oa_joined group by lad23, property_type where oa in ({codes_string});
    '''
  else:
    query = f'''
    with codes as (select geography_code from nssec_data order by rand () limit {random_number})
    select avg(price), stddev(price), oa21, property_type from pp_data_oa_joined where oa21 in (select * from codes) group by oa21, property_type;
    '''
  prices = query_to_dataframe(connection, query)
  prices['avg(price)'] = prices['avg(price)'].astype(float)
  prices['stddev(price)'] = prices['stddev(price)'].astype(float)

  prices = prices[prices['property_type'].isin(types)].drop('property_type',axis=1).groupby('lad23', as_index=False).mean()

  merged = df.merge(prices, left_on=label, right_on='oa21', how='left')
  med = merged['avg(price)'].median()
  return merged.fillna(med)