def create_nssec_table(conn):
    drop = "DROP TABLE IF EXISTS nssec_data"
    create_query = """
          CREATE TABLE IF NOT EXISTS `nssec_data` (
            census_date date NOT NULL,
            geography_code tinytext COLLATE utf8_bin NOT NULL,
            total_over_16 int(10) unsigned NOT NULL,
            L1_3 INT UNSIGNED NOT NULL COMMENT 'L1, L2 and L3 Higher managerial, administrative and professional occupations',
            L4_6 INT UNSIGNED NOT NULL COMMENT 'Lower managerial, administrative and professional occupations',
            L7 INT UNSIGNED NOT NULL COMMENT 'Intermediate occuptations',
            L8_9 INT UNSIGNED NOT NULL COMMENT 'Small employers and own account workers',
            L10_11 INT UNSIGNED NOT NULL COMMENT 'Lower supervisory and technical occupations',
            L12 INT UNSIGNED NOT NULL COMMENT 'Semi Routine',
            L13 INT UNSIGNED NOT NULL COMMENT 'Routine Occupations',
            L14 INT UNSIGNED NOT NULL COMMENT 'Never worked and long time unemployed',
            L15 INT UNSIGNED NOT NULL COMMENT 'Full time students',
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

    add_primary_key = "ALTER TABLE nssec_data ADD PRIMARY KEY (db_id)";
    auto_increment = "ALTER TABLE nssec_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";

    conn.cursor().execute(drop)
    conn.cursor().execute(create_query)
    conn.cursor().execute(add_primary_key)
    conn.cursor().execute(auto_increment)
    conn.commit()

# It might be preferable to have this seperated from create_nssec_table
# Create appropriate indices for the NSSEC table
def create_nssec_index(conn):
  index_geography_query = """CREATE INDEX nssec_geography_code USING HASH ON nssec_data (geography_code)"""
  index_student_query = """CREATE INDEX nssec_L15 USING HASH ON nssec_data (L15)"""
  index_date_query = """CREATE INDEX nssec_date USING HASH ON nssec_data (census_date)"""
  index_latlong_query = """CREATE INDEX nssec_latlong USING HASH ON nssec_data (latitude, longitude)"""
  conn.cursor().execute(index_geography_query)
  conn.cursor().execute(index_student_query)
  conn.cursor().execute(index_date_query)
  conn.cursor().execute(index_latlong_query)
  conn.commit()


# Create the oa_latlong table (maps output area to latlong), contains multiple years
def create_oa_latlong_table(conn):
  drop = "DROP TABLE IF EXISTS oa_latlong_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `oa_latlong_data` (
            census_date date NOT NULL,
            oa tinytext COLLATE utf8_bin NOT NULL,
            lsoa tinytext COLLATE utf8_bin NOT NULL,
            lsoa_name VARCHAR(100) NOT NULL,
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,
            shape_area decimal(20,8) NOT NULL,
            shape_length decimal(20,8) NOT NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE oa_latlong_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE oa_latlong_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()

# Could generalise these loads, but that might cause more issues than make life easier
def load_oa_coord_data_to_sql(conn, csv_file):
  load_query = f"""LOAD DATA LOCAL INFILE "{csv_file}" INTO TABLE `oa_latlong_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES STARTING BY '' TERMINATED BY '\n' IGNORE 1 LINES;"""
  conn.cursor().execute(load_query)
  conn.commit()

def create_oa_latlong_index(conn):
  index_oa_query = """CREATE INDEX oa_latlong_output_area USING HASH ON oa_latlong_data (oa)"""
  index_date_query = """CREATE INDEX oa_latlong_date USING HASH ON oa_latlong_data (census_date)"""
  index_latlong_query = """CREATE INDEX oa_latlong_latlong USING HASH ON oa_latlong_data (latitude, longitude)"""
  conn.cursor().execute(index_oa_query)
  conn.cursor().execute(index_date_query)
  conn.cursor().execute(index_latlong_query)
  conn.commit()


def create_household_cars_data(conn):
  drop = "DROP TABLE IF EXISTS household_vehicle_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `household_vehicle_data` (
            census_date date NOT NULL,
            geography_code tinytext COLLATE utf8_bin NOT NULL,
            total INT UNSIGNED NOT NULL,
            no_vehicle_count INT UNSIGNED NOT NULL,
            one_vehicle_count INT UNSIGNED NOT NULL,                      
            two_vehicle_count INT UNSIGNED NOT NULL,                      
            three_vehicle_count INT UNSIGNED NOT NULL,
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,                   
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE household_vehicle_data ADD PRIMARY KEY (db_id)";
  
  cols = ['no_vehicle_count', 'one_vehicle_count', 'two_vehicle_count', 'three_vehicle_count']
  
  base_ratio_query = "ALTER TABLE household_vehicle_data ADD COLUMN replace_ratio DOUBLE GENERATED ALWAYS AS (IF(total = 0, NULL, replace_count / total)) STORED;"
  add_ratio_0 = base_ratio_query.replace('replace','no_vehicle')
  add_ratio_1 = base_ratio_query.replace('replace','one_vehicle')
  add_ratio_2 = base_ratio_query.replace('replace','two_vehicle')
  add_ratio_3 = base_ratio_query.replace('replace','three_vehicle')
  add_ratio_queries = [add_ratio_0, add_ratio_1, add_ratio_2, add_ratio_3]

  auto_increment = "ALTER TABLE household_vehicle_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";

  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  for q in add_ratio_queries:
    conn.cursor().execute(q)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()

def create_household_vehicle_index(conn):
  index_geography_query = """CREATE INDEX household_vehicles_code USING HASH ON household_vehicle_data (geography_code)"""
  index_ratio_query = """CREATE INDEX household_vehicles_no_car_ratio USING HASH ON household_vehicle_data (no_vehicle_ratio)"""
  index_date_query = """CREATE INDEX household_vehicles_date USING HASH ON household_vehicle_data (census_date)"""
  index_latlong_query = """CREATE INDEX household_vehicles_latlong USING HASH ON household_vehicle_data (latitude, longitude)"""
  conn.cursor().execute(index_geography_query)
  conn.cursor().execute(index_ratio_query)
  conn.cursor().execute(index_date_query)
  conn.cursor().execute(index_latlong_query)
  conn.commit()

def load_census_data_to_sql(conn, csv_file, table):
  load_query = f"""LOAD DATA LOCAL INFILE "{csv_file}" INTO TABLE `{table}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES STARTING BY '' TERMINATED BY '\n' IGNORE 1 LINES;"""
  conn.cursor().execute(load_query)
  conn.commit()


# OSM
def create_building_tag_table(conn):
  drop = "DROP TABLE IF EXISTS building_tag_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `building_tag_data` (
            osm_date date NOT NULL,
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,
            tag VARCHAR(100) NOT NULL,
            extra VARCHAR(200) NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE building_tag_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE building_tag_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()