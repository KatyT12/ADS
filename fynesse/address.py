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
import osmnx as ox

def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0):
  distance = distance_km / 111.2
  north = latitude + distance/2
  south = latitude - distance/2
  west = longitude - distance/2
  east = longitude + distance/2
  return (north, south, east, west)

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



def download_purchases_at_location(conn, longitude, latitude, distance_km = 1, year_onwards = 2020, output_file='output_file.csv'):
  start_date = str(year_onwards) + "-01-01"
  lat2, lat1, long2, long1 = get_bounding_box(latitude, longitude, distance_km) # N S E W

  cur = conn.cursor()
  print(f'select * from pp_data inner join ( select postcode from postcode_data where latitude between {lat1} and {lat2} and longitude between {long1} and {long2}) as po on po.postcode = pp_data.postcode where pp_data.date_of_transfer >= "' + start_date + '";')
  cur.execute(f'select * from pp_data inner join ( select postcode from postcode_data where latitude between {lat1} and {lat2} and longitude between {long1} and {long2}) as po on po.postcode = pp_data.postcode where pp_data.date_of_transfer >= "' + start_date + '";')
  rows = cur.fetchall()

  with open(output_file, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(rows)