from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import csv
import pymysql
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import re
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import osmnx as ox
from .access import *
from .util import *
from .clustering import *
########################
# Plotting
def plot_correlation(joined, labels):

  features = joined_with_area[labels]

  fig, ax = plt.subplots()
  im = ax.matshow(features.corr())

  ax.set_xticks(np.arange(len(labels)))
  ax.set_xticklabels(labels, fontsize=14, rotation=45)

  ax.set_yticks(np.arange(len(labels)))
  ax.set_yticklabels(labels, fontsize=14, rotation=45)

  fig.colorbar(im, ax=ax)
  plt.show()


# Plot the correlation between l1 and l2, over some variable over_column
def plot_correlation_over_variable(joined, l1, l2, over_column):
  fig, ax = plt.subplots(figsize=(12,3))
  x_axis = np.array(joined[over_column].drop_duplicates().sort_values())
  y_axis = []
  for x in x_axis:
    p = joined[joined[over_column] == x][[l1,l2]].corr()
    y_axis.append(p[l2].iloc[0])
  plt.bar(x_axis, y_axis)
  ax.set_xticklabels(x_axis, fontsize=7, rotation=90)
  ax.set_title('Correlation for each postcode')
  plt.show()



# Postcode, price per metre
def plot_mean_price_area_ratio(joined, over_column):
  fig, ax = plt.subplots(figsize=(12,3))
  x_axis = np.array(joined[over_column].drop_duplicates())
  y_axis = []
  for x in x_axis:
    r = joined[joined[over_column] == x]
    #print(joined[over_column]['price'].mean())

    mn = (r['price']/r['area']).mean()

    p = r['price'].sum()/r['area'].sum()

    y_axis.append(mn)
  plt.bar(x_axis, y_axis)
  ax.set_xticklabels(x_axis, fontsize=7, rotation=90)
  ax.set_title('£ Per square metre')
  plt.show()
  return y_axis


# Plot the correlation bar chart between nimby data and other
def plot_prices_corr_barchart(house_prices, census_df, nimby_df):
  df = house_prices.merge(census_df, left_on='lad23', right_on='geography_code').merge(nimby_df, left_on='lad23', right_on='LAD23CD')

  df[df['property_type'].isin(['T', 'S', 'D'])]
  df['value'] = np.log(df['avg(price)'].astype(float)) * df['tenure:owned']

  types = df['property_type'].drop_duplicates()
  fig, ax = plt.subplots()
  values = []
  cols = ['rag','avg_rag_flats', 'avg_rag_housing', 'avg_rag_estate']

  for i, p in enumerate(types):
    filtered = df[df['property_type'] == p]
    corr_vals = -filtered[['value', *cols]].corr()[cols]
    for col in cols:
      values.append([corr_vals[col][0], col, p])

  corr_df = pd.DataFrame(values)
  corr_df.columns = ['value', 'rag','property_type']
  sns.barplot(data=v, x='rag', y='value', ax = ax, hue='property')

# Count points of interest near coordinates
def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """

    points_of_interest = ox.geometries_from_bbox(*get_bounding_box(latitude, longitude, distance_km), tags)
    pois_df = pd.DataFrame(points_of_interest)
    ret_counts = {}

    tags_with_specific = list(tags.keys())
    for t in tags_with_specific:
        if t in pois_df.columns:
          if isinstance(tags[t], list):
            for v in tags[t]:
              ret_counts[v] = ((pois_df[pois_df[t].notnull()])[t] == v).sum()
          else:
            ret_counts[t] = pois_df[t].notnull().sum()
        else:
            if isinstance(tags[t], list):
                for v in tags[t]:
                  ret_counts[v] = 0
            else:
              ret_counts[t] = 0
    return ret_counts

# Clean the extracted price_paid and and given OpenStreetMap dataframes
#   :warning: Removes all rows of price paid data with street NA
def clean_geo_pp_data(con, pp_data, *dataframes_to_clean):
  #

  add_column_names(con, 'pp_data', pp_data, ['Postcode 2'])
  pp_data['street'] = [x.lower() if isinstance(x, str) else math.nan for x in pp_data['street']]

  for df in dataframes_to_clean:
    df['addr:street'] = [x.lower().replace('\'','') if isinstance(x, str) else x  for x in df['addr:street']]

  con.commit()


# Plot the area and points of interest onto a graph
#   :param longitude
#   :param latitude
#   :param distance: distance in km of the box
#   :param tags
#   :param ax: Axis to plot on
def plot_pois_on_map(latitude, longitude, distance, tags, ax):
  n, s, e, w = get_bounding_box(latitude, longitude, distance)
  try:
    pois = ox.features_from_bbox(tags=tags, bbox=(w, s, e, n))
  except Exception:
    return
  if len(pois) == 0:
    return

  tags_with_specific = list(tags.keys())
  pois['map_color'] = [(0,0,0)] * len(pois)
  pois['map_label'] = 'None'
  colors = [(0,0,0)]
  labels = ['None']

  for t in tags_with_specific:
      if t in pois.columns:
        if isinstance(tags[t], list):
          for v in tags[t]:
            cond = (pois[t].notnull()) & (pois[t] == v)
            
            col = tuple(np.random.rand(3,))
            pois.loc[cond, 'map_color'] = pois[cond].apply(lambda _: col, axis=1)
            
            pois.loc[cond, 'map_label'] = v
            colors.append(col)
            labels.append(v)
        else:
          cond = pois[t].notnull()
          col = tuple(np.random.rand(3,))
          pois.loc[cond, 'map_label'] = t
          pois.loc[cond, 'map_color'] = pois[cond].apply(lambda _: col, axis=1)
          colors.append(col)
          labels.append(t)
  
  legend_handles = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]
  pois.plot(ax=ax, alpha = 0.7, color = pois['map_color'], label = pois['map_label'].to_numpy())
  ax.legend(handles=legend_handles)



def coord_to_nssec_data(connection, latitude, longitude, distance_km=0.5):
  n, s, e, w = access.get_bounding_box(latitude, longitude, distance_km)
  cur = connection.cursor()
  cur.execute(f'select geography_code, longitude, latitude, L15, total_over_16 from nssec_data where latitude between {s} and {n} and longitude between {w} and {e}')

  column_names = list(map(lambda x: x[0], cur.description))
  df = pd.DataFrame(columns=column_names, data=cur.fetchall())
  return df


# Plot the area and points of interest WITH student proportional population onto a graph
#   :param longitude
#   :param latitude
#   :param distance: distance in km of the box
#   :param tags
#   :param ax: Axis to plot on
def plot_location_students(connection, latitude, longitude, distance, location = '', with_labels=False, ax=None, tags={}):
  df = coord_to_nssec_data(connection, latitude, longitude, distance)
  df['student_proportion'] = df['L15']/df['total_over_16'].astype(float)

  n, s, e, w = access.get_bounding_box(latitude, longitude, distance)

  # Retrieve geographical features
  graph = ox.graph_from_bbox(bbox=(w, s, e, n))
  nodes, edges = ox.graph_to_gdfs(graph)

  # Split into percentiles
  number = 10
  green = np.array([0, 1, 0])
  red = np.array([1, 0, 0])
  colors = [ tuple(x) for x in np.linspace(green, red, number)]
  df['colors'] = pd.qcut(df['student_proportion'].to_numpy(), number, labels=colors)

  if ax is None:
    fig, ax = plt.subplots(figsize=(10,10))
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
  ax.scatter(df['longitude'], df['latitude'], 100, c=df['colors'], alpha=1.0, zorder=10)

  # Plot relevant buildings onto the map
  
  plot_pois_on_map(latitude, longitude, distance, tags, ax)
  
  # Annotate with output area labels
  if with_labels:
    for lat,lon, oa in df[['latitude', 'longitude', 'geography_code']].itertuples(index=False):
      ax.annotate(oa, (float(lon)+0.00001, float(lat)+0.00001))

  ax.set_title('Proportional student population mapped at ' + location)


#---------------------------------------------------- Code for querying training data

def query_to_dataframe(conn, query):
  cur = conn.cursor()
  cur.execute(query)
  column_names = [x[0] for x in cur.description]
  df = pd.DataFrame(columns=column_names, data=cur.fetchall())
  df = df.loc[:,~df.columns.duplicated()].copy()
  return df

def query_random_set(conn, number, table='household_vehicle_data', pred_table='nssec_data', dist_query=1):
  query = f'''
    with codes AS (
        select geography_code
        FROM code_count_table
        GROUP BY geography_code
        order by rand ()
        limit {number}
    )
    select *
    from code_count_table as cc
    join {table} as hv on hv.geography_code = cc.geography_code 
    join {pred_table} as ns on ns.geography_code = cc.geography_code 
    WHERE cc.geography_code IN (SELECT * FROM codes) and cc.distance = {dist_query};
  '''
  return query_to_dataframe(conn, query)

def random_query_table(conn, number, table):
  query = f'''
    select *
    from {table}
    order by rand ()
    limit {number}
  '''
  return query_to_dataframe(conn, query)

def query_training_for_location(conn, latitude, longitude, distance, table='household_vehicle_data', pred_table='nssec_data', dist_query=1):
  
  #lat_dist, lon_dist = fynesse.access.latlong_to_km(52.5152422, -1.1482686, distance, distance)
  n, s, e, w = access.get_bounding_box(latitude, longitude, distance)

  query = f'''
    with codes AS (
        select geography_code
        FROM {pred_table}
        where latitude between {s} and {n} and longitude between {w} and {e}
        GROUP BY geography_code
    )
    select *
    from code_count_table as cc
    join {table} as hv on hv.geography_code = cc.geography_code 
    join {pred_table} as ns on ns.geography_code = cc.geography_code 
    WHERE cc.geography_code IN (SELECT * FROM codes) and cc.distance = {dist_query};
  '''
  return query_to_dataframe(conn, query)


def map_new_build_areas(conn, year_from = 1995, year_to = 2024, property_types=['T', 'F', 'D', 'O', 'S'], threshold=5, groupings=None, by_lad= False):
  # Online join
  ptype_string = ', '.join(["'" + s + "'" for s in property_types])
  if not by_lad:
    query = f'select * from new_build_data as nb join postcode_oa_lad as pd on pd.postcode = nb.postcode where count > {threshold} and property_type in ({ptype_string}) and year between {year_from} and {year_to};'
  else:
    query = f'select * from new_build_data as nb join postcode_data as pd on pd.postcode = nb.postcode where count > {threshold} and property_type in ({ptype_string}) and year between {year_from} and {year_to};'

  df = query_to_dataframe(conn, query)
  

  
# Retrieve geopandas data for plotting
def retrieve_map_data(link = 'https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/3f29d2c4a5834360a540ff206718c4f2/shapefile?layers=0', dir='LAD_boundaries', file='LAD_DEC_2023_UK_BFE.shp'):
  shapefile_path = dir + '/' + file
  if not os.path.exists(shapefile_path):
    download_and_unzip(link, '', '', location=dir, all=True)
  gdf = gpd.read_file(shapefile_path)
  return gdf

# ------------------ PCA

def plot_pca(data, points, colour_col=None, ax=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(10,10))
  
  pca = PCA(n_components=2)
  transformed = pca.fit_transform(points)
  if colour_col is not None:
    ax.scatter(transformed[:,0], transformed[:,1], c=data[colour_col], alpha=0.7)
  else:
    ax.scatter(transformed[:,0], transformed[:,1], alpha=0.7)
  return pca

def get_cols(df, name):
  return df.columns[df.columns.str.contains(name)]

def get_components(points, num):
  pca = PCA(n_components=num)
  transformed = pca.fit_transform(points)
  pca.fit(points)
  return pca.components_

def get_features(points, num, base):
  pca = PCA(n_components=num)
  transformed = pca.fit_transform(points)
  features = pd.DataFrame(transformed)
  features['geography_code'] = base['geography_code']
  return features

def plot_components(points, num, cols, x_axis, ax = None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(23,8),  ncols=2)
  
  components = get_components(points, num)
  sns.heatmap(components, annot=True, cmap="coolwarm", xticklabels=cols, cbar=True, ax=ax[0])
  for i,c in enumerate(components):
    ax[1].plot(x_axis, c, label=f'PC {i}')
  ax[1].legend()
  ax[1].set_title('Principal components')



# Plot new builds per area, plot london seperately for visibility
def map_new_build_areas(conn, year_from = 1995, year_to = 2024, property_types=['T', 'F', 'D', 'O', 'S'], threshold=5, groupings=None, by_lad= False, ax = None, iqr=False):
  # Online join
  ptype_string = ', '.join(["'" + s + "'" for s in property_types])
  if by_lad:
    query = f'select lad23, lad21, lad_name, count(*) as number from new_build_data as nb join postcode_area_data as pd on pd.postcode = nb.postcode where count > {threshold} and property_type in ({ptype_string}) and year between {year_from} and {year_to} group by pd.lad23;'
  else:
    query = f'select * from new_build_data as nb join postcode_data as pd on pd.postcode = nb.postcode where count > {threshold} and property_type in ({ptype_string}) and year between {year_from} and {year_to};'


  if ax is None:
    fig, ax = plt.subplots()
  
  df = query_to_dataframe(conn, query)
  if by_lad:    
    gdf = retrieve_map_data()
    gdf = gdf.merge(df, left_on = 'LAD23CD', right_on='lad23')
    cmap = plt.cm.coolwarm

    if not iqr:
      norm = mcolors.Normalize(vmin=gdf['number'].min(), vmax=gdf['number'].max())
      gdf['normed'] = gdf['number'].apply(lambda x: norm(x))
    else:
      low, high = np.percentile(gdf['number'], [2, 98])
      gdf['normed'] = np.clip((gdf['number'] - low) / (high - low), 0, 1)
    
    gdf['col'] = gdf['normed'].apply(lambda x: mcolors.to_hex(cmap(x)))
    gdf.plot(edgecolor="black", color=gdf['col'], ax = ax)
    ax.set_xticks([])
    ax.set_yticks([])

    leg = [
      mpatches.Patch(color='blue', label='High number of builds'),
      mpatches.Patch(color='red', label='Low number of builds'),
    ]

    ax.legend(handles=leg)

    # Plot London
    lon = inset_axes(ax, width="20%", height="20%", loc="lower right") 
    lon.set_xticks([])
    lon.set_yticks([])

    london_lads = in_london(gdf)
    london_lads.plot(edgecolor="black", color=london_lads['col'], ax = lon)
  else:
    ax.scatter(df['longitude'], df['latitude'], color = 'red', s = 0.01, alpha=0.7)
    ax.set_title('Location of new builds')
    return df


# Query electoral data to lad
def get_electoral_to_lad_weighted(conn):
  query = f'''
  with tab as (select  pcon25, lad23, count(*) as count from cons_to_oa_data group by pcon25, lad23),
  counts as (select pcon25, count(*) as count from cons_to_oa_data group by pcon25)
  select tab.pcon25, tab.lad23, tab.count/counts.count as weight, ed.* from tab join counts on tab.pcon25 = counts.pcon25 join electoral_data as ed on ed.ons_id = tab.pcon25
  '''
  return query_to_dataframe(conn, query)

def retrieve_electral_lad_aggregated(connection):
  electoral_lad_df = get_electoral_to_lad_weighted(connection)
  # Multiply by weight
  electoral_lad_df.loc[:,['electorate','valid_votes','con','lab','ld','ruk','green']] = electoral_lad_df[['electorate','valid_votes','con','lab','ld','ruk','green']].multiply(electoral_lad_df['weight'], axis=0).astype(float)
  # Sum and group by LAD
  result = electoral_lad_df.groupby("lad23")[['electorate','valid_votes','con','lab','ld','ruk','green']].sum().reset_index()
  return result

#----------

# A generic method for plotting colours on a map of England/the UK
def plot_colours(gdf, df, key='LAD23CD', england_only=True, ax = None):
  merged = gdf.merge(df, left_on='LAD23CD', right_on='lad23')
  if england_only:
    merged = merged[merged['LAD23CD'].str.contains('E')]
  
  if ax is None:
    fig, ax = plt.subplots()
  merged.plot(color = merged['colours'], alpha=0.7, ax=ax)



# -------------- House prices
def get_avg_price(connection):
  query = '''
    select avg(price), stddev(price), lad23, property_type from pp_data_oa_joined group by lad23, property_type;
  '''
  return query_to_dataframe(connection, query)

def house_prices_against_rag(connection, nimby_df, remove_outliers=False):
  df = get_avg_price(connection)
  house_prices22 = df.merge(nimby_df, left_on='lad23', right_on='LAD23CD')
  price_by_p_type = {p_type: rest for p_type, rest in house_prices22.groupby('property_type')}

  fig, ax = plt.subplots(nrows=len(list(price_by_p_type.keys())), ncols=2, figsize=(14, 26))
  for i, (prop_type, df) in enumerate(list(price_by_p_type.items())):
    price_by_p_type

    
    # Use IQR to remove outliers
    if remove_outliers:
      low_q = df['avg(price)'].astype(float).quantile(0.25)
      high_q = df['avg(price)'].astype(float).quantile(0.75)
      quartile_range = high_q - low_q

      lower = low_q - 1.5 * quartile_range
      upper = high_q + 1.5 * quartile_range
      df = df[(df['avg(price)'].astype(float) >= lower) & (df['avg(price)'].astype(float) <= upper)]
    
    j = df['avg(price)'].argsort()
    price_by_p_type[prop_type] = df.iloc[j]

    n = len(df.index)
    ax[i][0].plot(np.arange(n), np.log(df.iloc[j]['avg(price)'].astype(float)), label=f'Average house price for {prop_type}')
    ax[i][0].scatter(np.arange(n), df.iloc[j]['rag']*3, label='Support for new builds 1-10 ', s=3, color = 'orange')
    ax[i][0].set_title(f'LAD ordered by {prop_type} price against avg(price) and support for new builds')
    ax[i][1].scatter(np.log(df['avg(price)'].astype(float)), df['rag'], s=3)

    ax[i][1].set_title(f'Log {prop_type} price against RAG')
    ax[i][1].set_xlabel(f'Log {prop_type} price')
    ax[i][1].set_ylabel(f'Support for new builds (RAG)')
    ax[i][1]
    ax[i][0].legend()


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
