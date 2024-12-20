from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""
import requests
import pymysql
import csv
import time
import os
import math
import pandas as pd
import numpy as np
import osmnx as ox
import osmium
import zipfile
import io
from functools import reduce

import yaml
import re
from ipywidgets import interact_manual, Text, Password
from .util import *
from .table_create import *

global CONNECTION
CONNECTION = None

# A function to download arbitrary CSV files
# :param url: url of the csv file
# :param file_name: name of the file to write to
def download_arbitrary_csv(url, file_name):
        file_name = file_name
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)


# Download and extract a file
def download_and_unzip(url, file_name, zip_name, location='.', all=False):
  dir = location
  os.makedirs(dir, exist_ok = True)

  response = requests.get(url)
  if response.status_code == 200:
      with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        if all:
          z.extractall(path=location)
        else:
          z.extract(zip_name, dir)
          os.rename(dir + '/' + zip_name, dir + '/' + file_name)
          

  if os.path.exists(dir) and os.listdir(dir):
    print(f"Files already exist at: {dir}.")
    return

# A function to create a connection to any SQL database
""" Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
"""
def create_connection(database='ads_2024', cred_file='credentials.yaml'):
    CONNECTION = None
    if CONNECTION == None or not connection.open:
        conn = None
        try:
          with open(cred_file) as file:
            credentials = yaml.safe_load(file)
            user = credentials["username"]
            password = credentials["password"]
            host = credentials["url"]
            port = int(credentials["port"])
            db = database
          try:
              conn = pymysql.connect(user=user,
                                  passwd=password,
                                  host=host,
                                  port=port,
                                  local_infile=1,
                                  db=database
                                  )
              print(f"Connection established!")
          except Exception as e:
              print(f"Error connecting to the MariaDB Server: {e}")
        except FileNotFoundError:
          print(f"Could not find {cred_file}, please call create_credentials() and enter your details first")
          return None
        
        CONNECTION = conn
        return conn
    else:
        return CONNECTION




# Download purchases at a box and write into a given csv file
#   :param user: connection
#   :param password: longitude
#   :param host: latitude
#   :param distance_km: distance in km

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


# Download purchases places in csv file and then returns a dataframe
#   :param user: connection
#   :param password: longitude
#   :param host: latitude
#   :param distance_km: distance in km
def load_as_dataframe(con, longitude, latitude, distance_km=2, year_onwards=1994):
  print(get_bounding_box(latitude, longitude, distance_km))

  download_purchases_at_location(con, longitude, latitude, distance_km=distance_km, year_onwards=year_onwards, output_file='1994_houses_bought.csv')

  data = pd.read_csv('./1994_houses_bought.csv')
  con.commit()
  return data

# Upload the jdata from joining postcodes and pp_data
#   :param connection: Database connection
#   :param year: year of transactions
def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))


# Add the appropriate column names to a dataframe taken from SQL, with additional columns possibly
#   :param conn: Database connection
#   :param table_name: Name of the table to take columns from
#   :param data_frame: Data frame to add columns to
#   :additional_columns: additional columns to add
#   :warning: will throw an error if the dimensions do not add up

def add_column_names(conn, table_name, data_frame , additional_columns = []):
    cur = conn.cursor()
    cur.execute('select column_name from information_schema.columns where table_name = "' + table_name + '";')
    output = cur.fetchall()
    cols = np.array(output)[:,0]
    data_frame.columns = list(cols.astype(str)) + additional_columns


#   Simply collect all points of interest in the given bounding box
#   :param longitude
#   :param latitude 
#   :param distance_km: width of square
def get_buildings_data_from_geo(latitude, longitude, tags, distance_km=2):
  n, s, e, w = get_bounding_box(latitude, longitude, distance_km)
  print(n,s,e,w)
  points_of_interest = ox.geometries_from_bbox(n, s, e, w, tags)
  return points_of_interest

#   Filter a dataframe based on if rows have each tag tags. Return a set of rows which do and the complement
#   :param df: the data frame
#   :param tags: tags to filter
def filter_dataframe(df, tags):
  ret = df
  ret_ind = df.index
  for t in tags:
    ret_ind = ret_ind.intersection(ret.loc[ret_ind][t].dropna().index)
  
  return ret.loc[ret_ind], ret.loc[~ret.index.isin(ret_ind)]


#   Get 'addressed' data which is rows that have a house number, street and optionally a postcode
#   :param longitude
#   :param latitude 
#   :param distance_km: width of square
#   :param with_postcode: include postcode in filtering
#   :param raw: dataframe or raw object from the API
#   :param tags: tags to search for
def get_addressed_buildings_data_from_geo(latitude, longitude, distance_km=2, with_postcode = False, raw=False, tags = {
    'building': True,
    'addr:housenumber':True,
    'addr:street':True,
    'addr:postcode':True,
  }):
 
  points_of_interest = get_buildings_data_from_geo(latitude, longitude, tags, distance_km)
  poi_df = pd.DataFrame(points_of_interest)
  poi_df['area'] = to_sqm(points_of_interest['geometry'])
  print(len(poi_df.index))

  if with_postcode:
    #buildings_with_addressses = points_of_interest[poi_df['addr:housenumber'].notnull() & poi_df['addr:street'].notnull() & poi_df['addr:postcode'].notnull()]
    buildings_with_addressses, buildings_without_addresses = filter_dataframe(poi_df, ['addr:housenumber', 'addr:street', 'addr:postcode'])
  else:
    buildings_with_addressses, buildings_without_addresses = filter_dataframe(poi_df, ['addr:housenumber', 'addr:street'])
  if raw:
    return buildings_with_addressses, buildings_without_addresses
  else:
    return pd.DataFrame(buildings_with_addressses), pd.DataFrame(buildings_without_addresses)





# Longitude and latitude convert

def latlong_to_km(latitude: float, longitude: float, lat_dist, lon_dist):
  distance_lat = lat_dist / 110.574
  distance_long = lon_dist / (math.cos(latitude * math.pi / 180) * 111.32)
  return (distance_lat, distance_long)

# Approximately convert square degree to squre metres


def to_sqm(geometry):
    return geometry.to_crs(6933).area


def latlong_diff_to_distance(latitude: float, longitude: float, lat_dist, lon_dist):
  distance_lat, distance_long = latlong_to_km(latitude, longitude, lat_dist, lon_dist)
  total_distance = sqrt(distance_lat**2 + distance_long**2)
  return total_distance

def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0):
  distance_lat, distance_long = latlong_to_km(latitude, longitude, distance_km, distance_km)

  north = latitude + (distance_lat/2)
  south = latitude - (distance_lat/2)
  west = longitude - (distance_long/2)
  east = longitude + (distance_long/2)
  return (north, south, east, west)



def create_credentials():
  @interact_manual(username=Text(description="Username:"),
                password=Password(description="Password:"),
                url=Text(description="URL:"),
                port=Text(description="Port:"))
  def write_credentials(username, password, url, port):
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username,
                           'password': password,
                           'url': url,
                           'port': port}
        yaml.dump(credentials_dict, file)


#****************************** Extracting, filtering and uploading OSM data to an SQl database

# Filter out a pbf file to just the tags of interest
def extract_features_to_file(tags, output_file='england-filtered.osm.pbf', input_file = 'england-latest.osm.pbf'):
  filter_string = ''
  for t in tags.keys():
    if isinstance(tags[t], list):
      for v in tags[t]:
        filter_string += f' "{t}={v}",'
    else:
      filter_string += f' "{t}=*",'
  filter_string = filter_string[:-1]

  command = f'osmium tags-filter {input_file} {filter_string} -o {output_file}'  
  print(f'Running: {command}')
  os.system(command)

# Uploader to read the PBF data and upload relevant tags
class StopProcessing(Exception):
  pass

class OSMUploader(osmium.SimpleHandler):
    decoded_num = 0
    heart_beat = 50000
    tags = ['addr', ]
    

    def __init__(self, tags, connection):
        super().__init__()
        self.tags = tags
        self.nodes = []
        self.ways = []
        self.connection = connection
        self.relations = []

    def upload_arr(self, arr, name):
      print(f"Uploading {len(arr)}")
      df_nodes = pd.DataFrame(arr)
      df_nodes.to_csv(name, index=False)
      load_census_data_to_sql(self.connection, name, 'building_tag_data')

    def decide_stop(self):
      self.decoded_num += 1
      if self.decoded_num > self.heart_beat:
        if len(self.nodes) > 0:
          self.upload_arr(self.nodes, 'nodes_extracted.csv')
        if len(self.ways) > 0:
          self.upload_arr(self.ways, 'ways_extracted.csv')
        self.decoded_num = 0
        self.nodes = []
        self.ways = []
        
        #raise StopProcessing

    def node(self, n):
        k = dict(n.tags).keys()
        d = dict(n.tags)
        if True:
          for t in self.tags:
            if isinstance(self.tags[t], list):
              if t in k and d[t] in self.tags[t]:
                self.decide_stop()
                #print("NODE", dict(n.tags))
                self.nodes.append(['2024-01-01', n.location.lat, n.location.lon, d[t], ''])
                break
            else:
              if t in k:
                self.decide_stop()
                self.nodes.append(['2024-01-01', n.location.lat, n.location.lon, t, ''])
                break
                #print("NODE", dict(n.tags))
          


    def way(self, w):
        # Check if the node lies within the radius (e.g., 0.001 degrees)
        k = dict(w.tags).keys()
        d = dict(w.tags)
        lat = -1
        lon = -1
        if len(w.nodes) > 0:
            lat_sum = 0
            lon_sum = 0
            count = 0
            for node in w.nodes:
              if node.location.valid():
                lat_sum += node.location.lat
                count+=1
                lon_sum += node.location.lon
            if count > 0:
              lat = lat_sum/count
              lon = lon_sum/count
        
        if True:
          for t in self.tags:
            if isinstance(self.tags[t], list):
              if t in k and d[t] in self.tags[t]:
                self.decide_stop()
                #print("WAY", dict(w.tags))
                self.ways.append(['2024-01-01', lat, lon, d[t], ''])
                break
            else:
              if t in k:
                self.decide_stop()
                self.ways.append(['2024-01-01', lat, lon, t, ''])
                #print("WAY", dict(w.tags))
                break 



def upload_with_handler(handler, file_name, tags):
  handler.apply_file(file_name, locations=True)
  if len(handler.nodes) > 0:
    handler.upload_arr(handler.nodes, 'nodes_extracted.csv')
  if len(handler.ways) > 0:
    handler.upload_arr(handler.ways, 'ways_extracted.csv')


def create_count_table(conn, distance, tags_excluded):
   name = 'code_count_table'
   drop = f"DROP TABLE IF EXISTS {name}"
   
   
   tags_list = [ f'\'{s}\'' for s in flatten_tags(tags_excluded)]
   tag_string = '(' + ', '.join(tags_list) + ')'

   lat_dist, lon_dist = latlong_to_km(52.5152422, -1.1482686, distance/2, distance/2)
   
   query1 = f'''
   create table {name} as
   select geography_code, tag, count(*) as count, {distance} as distance from nssec_data as a join building_tag_data as b on (b.latitude between a.latitude - {lat_dist} and a.latitude + {lat_dist} and b.longitude between a.longitude - {lon_dist} and a.longitude + {lon_dist})  group by geography_code, tag
   '''
   
   query = f'''
   create table {name} as
   select geography_code, tag, count(*) as count, {distance} as distance from nssec_data as a join (select * from building_tag_data where tag in {tag_string}) as b on (b.latitude between a.latitude - {lat_dist} and a.latitude + {lat_dist} and b.longitude between a.longitude - {lon_dist} and a.longitude + {lon_dist})  group by geography_code, tag
   '''
   print(query1)
   conn.cursor().execute(drop)
   conn.cursor().execute(query1)
   conn.commit()
  
def insert_into_count_table(conn, distance, tags_filter):
  lat_dist, lon_dist = latlong_to_km(52.5152422, -1.1482686, distance/2, distance/2)
  
  tags_list = [ f'\'{s}\'' for s in flatten_tags(tags_filter)]
  tag_string = '(' + ', '.join(tags_list) + ')'
  

  drop = '''drop temporary table if exists tagged_temporary;'''

  query1 = f'''
    create temporary table tagged_temporary AS
    SELECT *
    FROM building_tag_data
    WHERE tag in {tag_string};
  '''
  index = 'CREATE INDEX tagged_temporary_latlong USING HASH ON tagged_temporary (latitude, longitude);'
  

  print(drop)
  conn.cursor().execute(drop)
  print(query1)
  conn.cursor().execute(query1)
  print(index)
  conn.cursor().execute(index)
  
  upload = f'select geography_code, tag, count(*) as count, {distance} as distance from nssec_data as a join tagged_temporary as b on (b.latitude between a.latitude - {lat_dist} and a.latitude + {lat_dist} and b.longitude between a.longitude - {lon_dist} and a.longitude + {lon_dist})  group by geography_code, tag;'
  print(upload)

  cur = conn.cursor()
  cur.execute(upload)
  rows = cur.fetchall()
  csv_file_path = 'output_file.csv'

  print('Finished join')
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(rows)
  
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `code_count_table` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()


def get_nimby_data():
  if not os.path.exists('nimby_data.csv'):
    download_arbitrary_csv('https://datawrapper.dwcdn.net/fpMTl/8/dataset.csv', 'nimby_data.csv')
  nimby_df = pd.read_csv('nimby_data.csv')
  return nimby_df

# Census download, clean data
def remove_columns_from_df(csv_file, columns, path='.'):
    full = path + '/' + csv_file
    data = pd.read_csv(full)
    df = data.drop(columns, axis=1)
    return df

# Cleaning nssec data as specified, a base function for setting up any census data
def clean_census_data(csv_files, mapping_df, path='.'):
  for f in csv_files:
    full = path + '/' + f
    df = remove_columns_from_df(f, ['geography code'], path=path)
    df['date'] = df['date'].astype(str)
    df['date'] += '-01-01'

    # Add latitude and longitude data to avoid a join
    merged_df = pd.merge(df, mapping_df[['OA21CD', 'LAT', 'LONG']], left_on='geography', right_on='OA21CD', how='left')
    merged_df = merged_df.rename({'LAT': 'latitude', 'LONG': 'longitude'}, axis=1)
    merged_df = merged_df.drop('OA21CD', axis=1)

    merged_df.to_csv(full, index=False)

def clean_output_coords(csv_file, year, path='.'):
  df = remove_columns_from_df(csv_file, ['FID', 'BNG_E', 'BNG_N', 'LSOA21NMW', 'GlobalID'])
  # Add date
  df.insert(0, 'date', str(year) + '-01-01')
  df.to_csv(csv_file, index=False)


def download_census_data(c_type, code, years, base='.'):
  base_url = f'https://www.nomisweb.co.uk/output/census/year/census2021-{c_type.lower()}.zip'
  dir = os.path.join(base, 'census-'+c_type.lower())

  os.makedirs(dir, exist_ok = True)
  for year in years:
    url = base_url.replace('year', year)
    response = requests.get(url)

    if response.status_code == 200:
      file = os.path.splitext(os.path.basename(url))[0] + '-' + code + '.csv'
      with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extract(file, dir)
        os.rename(dir + '/' + file, f'{dir}/{year}-{code}.csv')

  if os.path.exists(dir) and os.listdir(dir):
    print(f"Files already exist at: {dir}.")
    return

def load_census_data(c_type, code='oa', year='2021'):
  return pd.read_csv(f'census-{c_type.lower()}/{year}-{code}.csv')


# Download and load census data, cache
def download_load(code, area='ltla', year='2021'):
  if not os.path.exists(f'census-{code}/{year}-{area}.csv'):
    download_census_data(code, area, years=[year])
  return load_census_data(c_type=code, code=area, year=year).drop(['date', 'geography'], axis=1)

# Clean out and aggregate the census composition daa appropriately
def clean_composition_data(df):
  ret = df
  ret['comp:total'] = df.iloc[:,1]
  ret['comp:single_family:total'] = df.iloc[:, 5]
  ret['comp:one_person:total'] = df.iloc[:, 2]
  ret['comp:single_family:dependent_children'] = df.iloc[:, 9] + df.iloc[:, 13] + df.iloc[:, 16]
  ret['comp:single_family:no_children'] = df.iloc[:, 8] + df.iloc[:, 12]
  ret['comp:single_family:non_dependant_children'] = df.iloc[:, 10] + df.iloc[:, 14] + df.iloc[:, 17]
  ret['comp:other:total'] = df.iloc[:, 18]

  return ret[['geography code','comp:total', 'comp:one_person:total', 'comp:single_family:total', 'comp:single_family:dependent_children', 'comp:single_family:no_children', 'comp:single_family:non_dependant_children', 'comp:other:total']]

# Retrieve the census data, and clean appropriately
def get_census_data(area, year):
  age_df = download_load('ts007a', area, year)
  composition_df = clean_composition_data(download_load('ts003', area, year))
  accomodation_type_df = download_load('ts044', area, year)
  deprivation_df = download_load('ts011', area, year)
  tenure_df = download_load('ts054', area, year)
  tenure_df = tenure_df.drop(tenure_df.columns[[3,4,5,6,8,9,11, 12]],axis=1)
  
  merged_df = reduce(lambda  left, right: pd.merge(left,right,on=['geography code'],how='outer'), [age_df, composition_df, accomodation_type_df, deprivation_df, tenure_df])
  merged_df.insert(0, 'date', year + '-01-01')
  return merged_df

