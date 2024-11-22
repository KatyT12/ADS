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
import math
import pandas as pd
import numpy as np
import osmnx as ox
import yaml
import re
from ipywidgets import interact_manual, Text, Password
from .util import *

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
            port = credentials["port"]
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



def download_price_paid_data(year_from, year_to):
        base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
        file_name = "/pp-<year>-part<part>.csv"

        for year in range(year_from, (year_to+1)):
            print(f"Downloading data for year: {year}")
            for part in range(1,3):
                url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
                response = requests.get(url)
                if response.status_code == 200:
                    with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                        file.write(response.content)


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


### Joins
#############
# Joins for the OpenStreetMap data and the normal data


#   Join based on matching the street and address
def exact_join(data, houses_df):
  joined1 = pd.merge(houses_df, data, left_on=['addr:street', 'addr:housenumber'], right_on=['street', 'primary_addressable_object_name'], how='inner')
  return joined1

#   Join based on matching the street and house number + housename
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
  geo_flats = houses_df[houses_df['addr:flats'].notnull()]   # Could change this
  chosen_flats = geo_flats[geo_flats['addr:flats'].map( lambda x: True if (re.search(r"^(\d+)-(\d+)$",x)) else False)]

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


def join_with_heuristics(data, houses_df):
  joined = pd.concat([exact_join(data, houses_df), join_house_names(data, houses_df)])
  join_on_with_flats(joined, houses_df, data)
  return joined


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

# Create tables
def create_nssec_table(conn):
    drop = "DROP TABLE IF EXISTS nssec_data"
    create_query = """
          CREATE TABLE IF NOT EXISTS `nssec_data` (
            geography_code tinytext COLLATE utf8_bin NOT NULL,
            total_over_16 int(10) unsigned NOT NULL,
            L1_3 INT UNSIGNED utf8_bin NOT NULL COMMENT 'L1, L2 and L3 Higher managerial, administrative and professional occupations',
            L4_6 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Lower managerial, administrative and professional occupations',
            L7 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Intermediate occuptations',
            L8_9 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Small employers and own account workers',
            L10_11 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Lower supervisory and technical occupations',
            L12 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Semi Routine',
            L13 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Routine Occupations',
            L14 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Never worked and long time unemployed',
            L15 INT UNSIGNED utf8_bin NOT NULL COMMENT 'Full time students',
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

    add_primary_key = "ALTER TABLE nssec_data ADD PRIMARY KEY (db_id)";
    auto_increment = "ALTER TABLE nssec_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";

    conn.cursor().execute(drop)
    conn.cursor().execute(create_query)
    conn.cursor().execute(add_primary_key)
    conn.cursor().execute(auto_increment)
    conn.commit()

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

