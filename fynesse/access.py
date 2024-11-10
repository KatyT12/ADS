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
import pandas as pd
import numpy as np
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
def create_connection(user, password, host, database, port=3306):
    CONNECTION = None
    if CONNECTION == None or not connection.open:
        conn = None
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



# Approximately convert square degree to squre metres
def to_sqm(deg):
    return deg * (111.2**2) * (1000**2)

# Retrieve buildings data from geographic data, specify tags and return the buildings with addresses, and without
def get_buildings_data_from_geo(latitude, longitude, distance_km=2, with_postcode = False, raw=False, tags = {
    'building': True,
    'addr:housenumber':True,
    'addr:street':True,
    'addr:postcode':True,
  }):
  n, s, e, w = get_bounding_box(latitude, longitude, distance_km)

  print(n,s,e,w)
  points_of_interest = ox.geometries_from_bbox(n, s, e, w, tags)
  poi_df = pd.DataFrame(points_of_interest)
  poi_df['area'] = to_sqm(points_of_interest['geometry'].area)

  if with_postcode:
    buildings_with_addressses = points_of_interest[poi_df['addr:housenumber'].notnull() & poi_df['addr:street'].notnull() & poi_df['addr:postcode'].notnull()]
    buildings_without_addresses = points_of_interest[~(poi_df['addr:housenumber'].notnull() & poi_df['addr:street'].notnull() & poi_df['addr:postcode'].notnull())]
  else:
    buildings_with_addressses = points_of_interest[poi_df['addr:housenumber'].notnull() & poi_df['addr:street'].notnull()]
    buildings_without_addresses = points_of_interest[~(poi_df['addr:housenumber'].notnull() & poi_df['addr:street'].notnull())]

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
def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

