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

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

# Practical 1 price data

def download_arbitrary_csv(url, file_name):
        file_name = file_name
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
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
    return conn



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


def add_column_names(conn, table_name, data_frame , additional_columns = []):
    cur = conn.cursor()
    cur.execute('select column_name from information_schema.columns where table_name = "' + table_name + '";')
    output = cur.fetchall()
    cols = np.array(output)[:,0]
    data_frame.columns = list(cols.astype(str)) + additional_columns


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

