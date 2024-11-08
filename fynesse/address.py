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

import csv
import pymysql
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import osmnx as ox

def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0):
  distance = distance_km / 111.2
  north = latitude + distance/2
  south = latitude - distance/2
  west = longitude - distance/2
  east = longitude + distance/2
  return (north, south, east, west)


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


def plot_area_vs_price(joined, over_column):
  fig, ax = plt.subplots(figsize=(12,3))
  x_axis = np.array(joined[over_column].drop_duplicates().sort_values())
  y_axis = []
  for x in x_axis:
    p = joined[joined[over_column] == x][['area','price']].corr()

    y_axis.append(p['price'].iloc[0])
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








#############
# Joins for the OpenStreetMap data and the normal data
def exact_join(data, houses_df):
  joined1 = pd.merge(houses_df, data, left_on=['addr:street', 'addr:housenumber'], right_on=['street', 'primary_addressable_object_name'], how='inner')
  return joined1

def join_house_names(data, houses_df):
    houses_df['housename_number_combined'] = ['' if isinstance(x['addr:housename'], float) else str(x['addr:housename'].upper()) + ', ' + x['addr:housenumber'] for _,x in houses_df.iterrows()]
    data_null_secondary = data[~data['secondary_addressable_object_name'].notnull()]
    joined2 =  pd.merge(houses_df, data_null_secondary, left_on=['addr:street', 'housename_number_combined'], right_on=['street', 'primary_addressable_object_name'], how='inner')
    return joined2

def join(data, houses_df):
  joined = pd.concat([exact_join(data, houses_df), join_house_names(data, houses_df)])
  return joined




###############
## Seperate Flat rows
def seperate_flats_data(houses_df):
  geo_flats = houses_df[houses_df['addr:flats'].notnull()]
  chosen_flats = geo_flats[geo_flats['addr:flats'].map( lambda x: True if (re.search(r"(\d+)-(\d+)",x)) else False)]

  chosen_flats['flat_num'] = pd.NA
  flats_nums = chosen_flats['addr:flats'].map(lambda x: x.split('-'))
  print(list(flats_nums))

  temp = chosen_flats.copy()
  for i,r in temp.iterrows():
    print(flats_nums[i])
    start = int(flats_nums[i][0])
    end = int(flats_nums[i][1])
    num = end - start + 1
    for j in range(start, end+1):
      new_row = r.copy()
      new_row['flat_num'] = 'FLAT ' + str(j)
      new_row['area'] = new_row['area']/num

      chosen_flats.loc[len(chosen_flats)] = new_row

  houses_df['flat_num'] = pd.NA
  houses_df = pd.concat([houses_df, chosen_flats[chosen_flats['flat_num'].notnull()]])
  return houses_df

def join_on_with_flats(joined, houses_df, data):
  flats = data[data['secondary_addressable_object_name'].map(lambda x: False if isinstance(x, float) else 'FLAT' in x) ]
  houses_df = seperate_flats_data(houses_df)
  houses_df['first_address'] = [ x['addr:housenumber'] if (x['housename_number_combined'] == '') else x['housename_number_combined'] for _,x in houses_df.iterrows()]
  joined_flats =  pd.merge(houses_df, flats, left_on=['addr:street', 'first_address', 'flat_num'], right_on=['street', 'primary_addressable_object_name', 'secondary_addressable_object_name'], how='inner')
  joined = pd.concat([joined, joined_flats])

  # Sanity check
  #data[data['date_of_transfer'].eq('1995-07-07') & data['street'].eq('bateman street')]

  return joined
