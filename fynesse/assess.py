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
import re
import matplotlib.pyplot as plt
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
#   :param place_name
#   :param pois_and_colours, the points of interest you want to include, and their colour on the map
def plot_location(longitude, latitude, distance, place_name, *pois_and_colours):
  n, s, e, w = get_bounding_box(latitude, longitude, 2)
  graph = ox.graph_from_bbox(n, s, e, w, tags)
  nodes, edges = ox.graph_to_gdfs(graph)

  area = ox.geocode_to_gdf(place_name.lower())

  fig, ax = plt.subplots(figsize=(16, 8))
  area.plot(ax=ax, facecolor="white")
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([w, e])
  ax.set_ylim([s, n])

  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")


  for (x, c) in pois_and_colours:
    
    x.plot(ax=ax, color = c, alpha = 0.7)
  plt.tight_layout()
  plt.show()


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
