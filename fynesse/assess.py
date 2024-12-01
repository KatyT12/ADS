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
import osmnx as ox
from .access import *
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

def query_random_set(conn, number, table='household_vehicle_data', pred_table='nssec_data'):
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
    WHERE cc.geography_code IN (SELECT * FROM codes);
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

def query_training_for_location(conn, latitude, longitude, distance, table='household_vehicle_data', pred_table='nssec_data'):
  
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
    WHERE cc.geography_code IN (SELECT * FROM codes);
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
def retrieve_map_data(dir='LAD_boundaries', file='LAD_DEC_2021_UK_BGC.shp'):
  download_and_unzip("https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/7ceb69f99a024752b97ddac6b0323ab0/shapefile?layers=0", '', '', location=dir, all=True)
  shapefile_path = dir + '/' + file
  gdf = gpd.read_file(shapefile_path)
  return gdf



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
