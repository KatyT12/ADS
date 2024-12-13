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
from .util import *
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec
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
  df = query_random_set(connection, number)
  training = extract_training_data(df)
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
  prices = query_to_dataframe(connection, query)
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
def get_avg_price_lad(connection, df, types, year_start=2023, year_end=2024):
  prices = get_avg_price(connection, year_start=2023, year_end=2024)
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
  prices = query_to_dataframe(connection, query)
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


def fit_model_OLS(connection, training, actual, t, design_func, augmented=None, alpha=None, reg_weight=0):
  if augmented is None:
    train = augment_training(training, actual)
    augmented = get_avg_price_lad(connection, train, ['T'])


  design = design_func(augmented, augmented)
  
  model = sm.OLS(augmented[t], design)
  

  if alpha is not None:
    fitted_model = model.fit_regularized(alpha=alpha, L1_wt=reg_weight)
    return fitted_model
  else:
    fitted_model = model.fit()
    return fitted_model

# For a given label, training set and output area data, fit the model and predict for the output area set, with regularisation
def get_prediction(connection, oa_data, training, design_func, label='rag', model=None, alpha=0.00014):
  if model is None:
    model = fit_model_OLS(connection, training, training, label, design_func, augmented=training, alpha=alpha, reg_weight=1)
  X = design_func(oa_data, training)
  pred = model.predict(X)
  pred_df = oa_data.copy()
  pred_df['prediction'] = pred
  return pred_df.drop_duplicates()

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



def split_data(k, n):
  ind = np.random.choice(range(n), size=(n,), replace=False)
  k_sets = np.array_split(ind, k)

  return k_sets

# Retrieve the training data from the splitted indices
def get_train(splitted, d, i):

  left = [d.iloc[splitted[j]] for j in range(i)]
  right = [d.iloc[splitted[j]] for j in range(i+1,len(splitted))]
  if i == 0:
      return pd.concat(right)
  elif i == len(splitted)-1:
      return pd.concat(left)
  else:
    return pd.concat([pd.concat(left), pd.concat(right)])

  total = np.concatenate(left, right)
  return total

def rmse_calc(predicted, actual):
  return np.sqrt(((predicted - actual)**2).mean())


# Training data - augmented, with resposne
# Subset of training data
def run_on_lad_subset(connection, training, subset_training, subset_test, response_col, design_func, regularized = False, alpha = 0.1, weight = 1.0):
  # Train
  
  if regularized:
    model = fit_model_OLS(connection, subset_training, subset_training, response_col, design_func, augmented=subset_training, alpha=alpha, reg_weight=weight)
  else:
    model = fit_model_OLS(connection, subset_training, subset_training, response_col, design_func, augmented=subset_training)
  
  X = design_func(subset_test, training)
  prediction = model.predict(X)
  actual = subset_test[response_col]

  # Extract data
  c = pd.DataFrame({
      "actual": actual,
      "predicted": prediction
      })

  corr = c.corr()['actual'].iloc[1]
  r2 = r2_score(actual, prediction)
  rmse = rmse_calc(prediction, actual)

  #print(corr, rmse, r2)
  return [corr, rmse, r2]


def get_k_folded_results(connection, k, training, response, design_func, regularized = False, alpha = 0.01, weight = 1.0):
  n = len(training.index)
  splitted = split_data(k, n)

  ret = []
  for i in range(len(splitted)):
    training_data = get_train(splitted, training, i)
    test_data = training.iloc[splitted[i]]

    #print(age_df.shape, a.shape, b.shape)
    ret.append(run_on_lad_subset(connection, training, training_data, test_data, response, design_func, regularized=regularized, alpha=alpha, weight=weight))

  dat = pd.DataFrame(ret)
  dat.columns = ['corr', 'rmse', 'r2']
  return dat

def retrieve_best_alpha(connection, training, response, weight, design_func, test_range=(-8,-2), cross_val_groups=10, number=100, linear=False):
  if linear:
    values = [0,*np.linspace(test_range[0],test_range[1],number)]
  else:
    values = [0,*np.logspace(test_range[0],test_range[1],number)]
  
  outputs = []
  for v in values:
    ret = get_k_folded_results(connection, cross_val_groups, training, response, design_func, regularized = True, alpha = v, weight= weight)
    outputs.append(ret.mean(axis=0).to_frame().T)
  results = pd.concat(outputs)
  results['alpha'] = values
  return results.reset_index()



  #--------------- Downscaling average price

def get_loc(connection, oa):
  query = f'''
  select latitude, longitude from nssec_data where geography_code = '{oa}';
  '''
  d = query_to_dataframe(connection, query)
  d = d.astype(float)
  return (d.iloc[0]['latitude'], d.iloc[0]['longitude'])

def get_locations(connection, oas):
  oa_string = ', '.join(["'" + s + "'" for s in oas])
  query = f'''
  select geography_code, latitude, longitude from nssec_data where geography_code in ({oa_string});
  '''
  d = query_to_dataframe(connection, query)
  d['latitude'] = d['latitude'].astype(float)
  d['longitude'] = d['longitude'].astype(float)
  return d

# Retrieve relevant price paid entries
def get_pp_entries_ordered(connection, oas, property_types, year_start=2023, year_end=2024):
  types_string = ', '.join(["'" + s + "'" for s in property_types])
  oa_string = ', '.join(["'" + s + "'" for s in oas])
  query = f'''
    select property_type, price, oa21 from pp_data_oa_joined where oa21 in ({oa_string}) and property_type in ({types_string}) and date_of_transfer between "{year_start}-01-01" and "{year_end}-01-01"
    order by field(oa21, {oa_string})
  '''
  return query_to_dataframe(connection, query)


def find_median_price_oa(connection, oa, types, number_search=20, number_med=10, default=200000, year_start=2023, year_end=2024):
  latlong = get_loc(connection, oa)
  nearest = find_nearest_output_areas(connection, *latlong, number_search)
  entries = get_pp_entries_ordered(connection, nearest['geography_code'], types, year_start=year_start, year_end=year_end)
  if len(entries.index) < number_med:
    return default # Default value if not enough found
  else:
    return entries.iloc[:number_med]['price'].median()

def join_oa_census_with_pp(connection, census_data, oas, all=False, year_start=2023, year_end=2024):
  query = ''
  if all:
    query = f'''
      select * from oa_median_price_data where year_start = {year_start} and year_end={year_end}
    '''
  else:
    oa_string = ', '.join(["'" + s + "'" for s in oas])
    query = f'''
      select * from oa_median_price_data where year_start = {year_start} and year_end={year_end} where geography_code in ({oa_string})
    '''
  dat = query_to_dataframe(connection, query)
  ret = census_data.merge(dat[['geography_code', 'median_price']], left_on='geography_code', right_on='geography_code')
  ret['avg(price)'] = ret['median_price'].astype(float)
  return ret

# Upload median price for output areas
# Either random output areas, or output areas for a specific location
def upload_oa_median_price(connection, random_number = None, lad = None, year_start=2022, year_end=2024, p_types=['T']):
  
  if random_number is None:
    query = f'''
    select oa21 from oa21_to_lad23_data where lad23 = '{lad}'
    '''
    codes = query_to_dataframe(connection, query)['oa21']
  else:
    codes = random_query_table(connection, number = random_number, table='nssec_data')['geography_code']
  
  df = {}
  count = 0
  for c in codes:
    print(c, count)
    df[c] = find_median_price_oa(connection, c, p_types, year_start=year_start, year_end=year_end)    
    count+=1
  
  rows = [ {'code': k, 'year_start': year_start, 'year_end': year_end, 'median': v, 'property_types': 'T'} for k,v in df.items()]
  pd.DataFrame(rows).to_csv('temp.csv',index=False)
  load_census_data_to_sql(connection, 'temp.csv', 'oa_median_price_data')



#----------- Simple model

def retrieve_lad23_for_oa(connection, output_areas):
  output_areas_string = ', '.join(["'" + s + "'" for s in output_areas])
  query = f'''
    select oa21, lad23 from oa21_to_lad23_data where oa21 in ({output_areas_string})
  '''
  return query_to_dataframe(connection, query)


def simplest_predict(connection, nimby_df, output_areas, response='rag', code_label='LAD23CD'):
  lad_mapping = retrieve_lad23_for_oa(connection, output_areas)
  return nimby_df.merge(lad_mapping, left_on= code_label,right_on='lad23')[['oa21','LAD23CD',response]]
#---

#-------------- Comparison plots
# Comparison between the simple model and given model for a given LAD location
def plot_location(connection, comparison, label, points = None, location=''):
  fig, ax = plt.subplots(ncols=3, figsize=(18,7))
  fig.suptitle(f'{location} prediction comparison', fontsize=16)
  # Plot histogram
  comparison['prediction'].hist(bins=20, ax=ax[0], alpha=0.7, edgecolor='black')
  ax[0].axvline(x=comparison[label][0], color='red', label='LAD score')
  ax[0].legend()
  ax[0].set_title('Histogram of output area predictions')

  # Plot differences
  get_colour_outliers(comparison, 10, (3,3), 'diff')
  
  # Map data for output areas
  gdf = retrieve_map_data(link= 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Output_Areas_2021_EW_BFC_V8/FeatureServer/replicafilescache/Output_Areas_2021_EW_BFC_V8_-9054797207862162063.zip', dir='OA_boundaries', file='OA_2021_EW_BFC_V8.shp')

  gdf = gdf[gdf['OA21CD'].isin(comparison['oa21'].drop_duplicates())]
  gdf = gdf.to_crs(epsg=4326)

  m = gdf.merge(comparison, left_on='OA21CD',right_on='oa21')
  m.plot(ax=ax[1], color=comparison['colours'])
  ax[1].set_title('Output area differences')
  leg = [
        mpatches.Patch(color='grey', label='Insignificant difference'),
        mpatches.Patch(color='blue', label='Prediction much lower than LAD'),
        mpatches.Patch(color='red', label='Prediction much higher than LAD'),
  ]
  ax[1].legend(handles=leg)

  # Population density map
  ax[2].set_title('Population density map')
  census = get_census_data('oa', '2021')
  census.columns = get_census_combined_cols() # Consistency with SQL columns
  
  with_total_pop = m[['oa21', 'geometry']].merge(census[['age:total', 'geography_code']], left_on='oa21', right_on='geography_code')
  with_total_pop['value'] = with_total_pop['age:total']/with_total_pop['geometry'].area
  get_colour_percentiles(with_total_pop, 10, (0,0,1), (1,0,0), 'value')
  
  if points is not None:
    ax[1].scatter(points[:,1], points[:,0],c='white', marker='x')

  m.plot(ax=ax[2], color=with_total_pop['colours'])


#------------- General comparison
def compare_models(connection, train1, train2, design1, design2, cols = ['rag', 'avg_rag_flats', 'avg_rag_housing', 'avg_rag_estate']):
  results = []
  # Try both models
  cols = ['rag', 'avg_rag_flats', 'avg_rag_housing', 'avg_rag_estate']
  for c in cols:
    results.append(get_k_folded_results(connection, 12, train1, c, design1).mean(axis=0).to_frame().T)
    results.append(get_k_folded_results(connection, 12, train2, c, design2).mean(axis=0).to_frame().T)
    
  # Compare results
  ret = pd.concat(results).set_index(pd.Index(['Model 1', 'Model 2']*len(cols)))
  ret['variable'] = cols + cols
  return ret

#--------------------
# Comparison plots and calculations between the simple model and the given model
def compare_pred_to_simple(connection, oa_data, training, nimby_df, model, design_func, model_title='Given model', label='rag', location=None, points=None):
  X = design_func(oa_data, training)
  pred = model.predict(X)
  pred_df = oa_data.copy()
  pred_df['prediction'] = pred
  simple_predict = simplest_predict(connection, nimby_df, oa_data['geography_code'], response=label)
  comparison = simple_predict.merge(pred_df, left_on='oa21', right_on='geography_code')
  comparison['diff'] = comparison['prediction'] - comparison[label]
  
  # merge with longlat
  comparison = comparison.merge(get_locations(connection, comparison['oa21']), left_on='oa21', right_on='geography_code')

  if location is None:
    # Plot General scatter plot
    fig, ax = plt.subplots(ncols=2, figsize=(15,7))
    
    ax[0].scatter(comparison[label], comparison['prediction'], s =3)
    ax[0].set_xlabel('Simple model prediction (LAD23)')
    ax[0].set_ylabel(f'{model_title} prediction')
    ax[0].set_title('Prediction comparison')


    # Plot geographic differences plot
    get_colour_outliers(comparison, 10, (2,2), 'diff', alpha=(0.3, 0.7))
    ax[1].scatter(comparison['longitude'], comparison['latitude'], s = 4, c=comparison['colours'], alpha=comparison['alpha'])

    # LAD boundaries
    gdf = retrieve_map_data()
    gdf = gdf.to_crs(epsg=4326)
    gdf[gdf['LAD23CD'].str.contains('E')].plot(alpha=0.2,ax=ax[1],edgecolor='dimgray')

    leg = [
        mpatches.Patch(color='grey', label='Insignificant difference'),
        mpatches.Patch(color='blue', label='Prediction much lower than LAD'),
        mpatches.Patch(color='red', label='Prediction much higher than LAD'),
    ]
    
    
    lon = inset_axes(ax[1], width="30%", height="30%", loc="center left") 
    lon.set_xticks([])
    lon.set_yticks([])

    london_lads = in_london(gdf, name_col='LAD23NM')
    
    london_lads.plot(alpha=0.2,ax=lon,edgecolor='dimgray')
    df = london_lads.merge(comparison, left_on='LAD23CD', right_on='LAD23CD')
    lon.scatter(df['longitude'], df['latitude'], s = 4, c=df['colours'], alpha=df['alpha'])
    
    ax[1].legend(handles=leg)
    ax[1].set_title('Geographic comparison (based on percentiles)')

  else:
   plot_location(connection, comparison, label, points=points, location=location) 

  return comparison



def aggregate_predictions(connection, oa_data, training, nimby_df, design_func, label='rag', plot=True):
  predictions = get_prediction(connection, oa_data, training, design_func, label=label)
  
  # Retrieve Lad23
  lad_mapping = retrieve_lad23_for_oa(connection, predictions['geography_code'])
  with_lad = predictions.merge(lad_mapping, left_on='geography_code', right_on='oa21')
  # Merge with ground truth and geo data
  aggregated_predictions = with_lad.groupby('lad23', as_index=False)['prediction'].mean()
  comparison = aggregated_predictions.merge(nimby_df, left_on='lad23', right_on='LAD23CD')
  comparison['diff'] = comparison['prediction'] - comparison[label]

  # Aggregate
  if plot:
    gdf = retrieve_map_data()
    
    
    gdf_augmented = gdf.merge(comparison, left_on='LAD23CD', right_on='lad23')
  
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(2, 3)  # 2 rows, 3 columns

    # Top row (3 plots)
    ax1 = fig.add_subplot(gs[0, 0])  # First plot
    ax2 = fig.add_subplot(gs[0, 1])  # Second plot
    ax3 = fig.add_subplot(gs[0, 2])  # Third plot
    ax = [ax1, ax2, ax3]

    # Bottom row (1 plot spanning all 3 columns)
    ax4 = fig.add_subplot(gs[1, :]) 

    get_colour_percentiles(gdf_augmented, 40, (1,0,0), (0,0,1), 'prediction', col_name='pred_colours')
    get_colour_percentiles(gdf_augmented, 40, (1,0,0), (0,0,1), label, col_name='ground_colours')
    get_colour_outliers(gdf_augmented, 10, (3,3), 'diff', low='blue', high = 'red', col_name='diff_colours')
    
    leg1 = [
        mpatches.Patch(color='blue', label=f'High {label}'),
        mpatches.Patch(color='red', label=f'Low {label}'),
    ]

    leg2 = [
        mpatches.Patch(color='blue', label=f'Aggregated prediction lower than ground truth'),
        mpatches.Patch(color='red', label=f'Aggregated prediction higher than ground truth'),
    ]

    

    gdf_augmented.plot(color = gdf_augmented['pred_colours'], ax = ax[0])
    ax[0].set_title(f'Aggregated prediction of {label}')
    gdf_augmented.plot(color = gdf_augmented['ground_colours'], ax = ax[1])
    ax[1].set_title(f'Ground truth of {label}')
    ax[0].legend(handles=leg1)
    ax[1].legend(handles=leg1)
    gdf_augmented.plot(color = gdf_augmented['diff_colours'], ax = ax[2])
    ax[2].set_title('Largest differences')
    ax[2].legend(handles=leg2)

    ax4.scatter(comparison[label],comparison['prediction'], s = 5)

    # Plot differences on a graph
    comparison['res_start'] = comparison[['prediction',label]].min(axis=1)
    comparison['res_end'] = comparison[['prediction',label]].max(axis=1)

    ax4.vlines(comparison[label], ymin=comparison['res_start'], ymax=comparison['res_end'], color='red', linestyle='dotted', label='Residual Lines', alpha=0.6)
    
    x = np.linspace(comparison['res_start'].min(), comparison['res_end'].max(), 10)
    ax4.plot(x, x, color='black', alpha=0.6)
    ax4.set_title(f'Residual plot of aggregated predicted {label} vs actual averaged {label}')
    ax4.set_xlabel('Ground truth')
    ax4.set_ylabel('aggregated prediction')
  
  corr = {'correlation':[comparison[['prediction', label]].corr()[label].iloc[0]], 'variable':label, 'avg_absolute_difference': comparison['diff'].abs().mean()}
  return pd.DataFrame(corr)

def compare_location(connection, lad, train, nimby_df, census_oa, model, design_func, points=None, location=''):
  codes = get_codes_from_lad(connection, lad=lad)
  census_oa['oa21'] = census_oa['geography_code']
  oa_data = join_oa_census_with_pp(connection, census_oa[census_oa['geography_code'].isin(codes)], [], all=True, year_start=2022, year_end=2024).drop('oa21', axis=1).drop_duplicates()
  comparison = compare_pred_to_simple(connection, oa_data, train, nimby_df, model, design_func, model_title='Model 2', location=location, points=points)
  return comparison


#------------------------ Assessing model methods

def compare_feature_correlations_from_prediction(connection, oa_data, training, design_func, model_name = 'Model 1', label='rag', alpha=0.00014):
  
  design_ltla = design_func(training, training)
  design_oa = design_func(oa_data, training)

  regularised_rag_model = fit_model_OLS(connection, training, training, 'rag', design_func, augmented=training, alpha=0.00014, reg_weight=1)
  params = regularised_rag_model.params.to_frame().T
  
  for c in design_ltla.columns:
    if params[c].iloc[0] == 0:
      design_ltla = design_ltla.drop(c, axis=1)  
      design_oa = design_oa.drop(c, axis=1)  
  
  correlations = []
  for i in range(1,len(design_ltla.columns)):
    col = design_ltla.columns[i]
    # Make predictions
    design_ltla_temp = design_ltla.drop(col, axis=1)
    design_oa_temp = design_oa.drop(col, axis=1)
    
    # Train
    model = sm.OLS(training[label], design_ltla_temp)
    fitted_model = model.fit_regularized(alpha=alpha, L1_wt=1)
    
    # Retrieve OA predictions
    predictions = fitted_model.predict(design_oa_temp)
    design_oa_temp['prediction'] = predictions

    # Use the predictions to predict the missing feature
    correlations.append(pd.DataFrame({'feature': [col], 'correlation': [np.corrcoef(predictions, design_oa[col])[0][1]], 'training_correlation': [np.corrcoef(design_ltla[col], training[label])[0][1]], 'model': model_name}))
    
  return pd.concat(correlations)


def plot_correlation_prediction_comparison(connection, oa_data, training, design_func1, design_func2, model_names):
  df = pd.concat(
    [compare_feature_correlations_from_prediction(connection, oa_data, training, design_func=design_func1, model_name=model_names[0]),
    compare_feature_correlations_from_prediction(connection, oa_data, training, design_func=design_func2, model_name =model_names[1])
    ])
  
  fig, ax = plt.subplots()
  x_positions = np.arange(len(df['feature'].drop_duplicates()))

  m1 = df[df['model'] == model_names[0]]
  m2 = df[df['model'] == model_names[1]]

  ax.scatter(x_positions, m1['training_correlation'], color='black')

  ax.axhline(y=0, linestyle='--', color='black')
  for x in x_positions:
    ax.axvline(x=x,linestyle='--', color='grey', alpha=0.2)

  for (m,c),l in zip(zip([m1,m2], ['blue', 'red']), labs):
    ax.scatter(x_positions[:len(m.index)], m['correlation'], color=c, alpha=0.4)
    ax.vlines(x_positions[:len(m.index)], ymin=m[['correlation','training_correlation']].min(axis=1), ymax=m[['correlation','training_correlation']].max(axis=1), color=c, alpha=0.4, label=l)

  ax.legend()
  ax.set_xticks(x_positions)
  ax.set_xticklabels(df['feature'].drop_duplicates(), rotation=45)