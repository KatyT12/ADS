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



def create_tenure_data(conn):
  drop = "DROP TABLE IF EXISTS tenure_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `tenure_data` (
            census_date date NOT NULL,
            geography_code tinytext COLLATE utf8_bin NOT NULL,
            total INT UNSIGNED NOT NULL,
            count_owned INT UNSIGNED NOT NULL,
            count_social_rented INT UNSIGNED NOT NULL,
            count_private_rented INT UNSIGNED NOT NULL,
            count_private_agency INT UNSIGNED NOT NULL,
            count_private_other INT UNSIGNED NOT NULL,
            count_rent_free INT UNSIGNED NOT NULL,
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,                   
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE tenure_data ADD PRIMARY KEY (db_id)";
  
  auto_increment = "ALTER TABLE tenure_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  

  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()


def create_transport_data(conn):
  drop = "DROP TABLE IF EXISTS transport_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `transport_data` (
            census_date date NOT NULL,
            geography_code tinytext COLLATE utf8_bin NOT NULL,
            total INT UNSIGNED NOT NULL,
            public_tranport_count INT UNSIGNED NOT NULL,
            cycle_count INT UNSIGNED NOT NULL,
            foot_count INT UNSIGNED NOT NULL,
            other_count INT UNSIGNED NOT NULL,
            latitude decimal(11,8) NOT NULL,
            longitude decimal(10,8) NOT NULL,                   
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE transport_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE transport_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";

  ratio_query = "ALTER TABLE transport_data ADD COLUMN cycle_ratio DOUBLE GENERATED ALWAYS AS (IF(total = 0, NULL, cycle_count / total)) STORED;"


  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(ratio_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()


def create_transport_index(conn):
  index_geography_query = """CREATE INDEX transport_code USING HASH ON transport_data (geography_code)"""
  index_date_query = """CREATE INDEX transport_date USING HASH ON transport_data (census_date)"""
  index_latlong_query = """CREATE INDEX transport_latlong USING HASH ON transport_data (latitude, longitude)"""
  conn.cursor().execute(index_geography_query)
  conn.cursor().execute(index_date_query)
  conn.cursor().execute(index_latlong_query)
  conn.commit()


# Load to the NSSEC table
def load_nssec_to_sql(conn, csv_file):
    load_query = f"""LOAD DATA LOCAL INFILE "{csv_file}" INTO TABLE `nssec_data` FIELDS TERMINATED BY ',' LINES STARTING BY '' TERMINATED BY '\n' IGNORE 1 LINES;"""
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


def create_building_tag_index(conn):
  index_date_query = """CREATE INDEX building_tag_date USING HASH ON building_tag_data (osm_date)"""
  index_latlong_query = """CREATE INDEX building_tag_latlong USING HASH ON building_tag_data (latitude, longitude)"""
  index_tag_query = """CREATE INDEX building_tag_tag USING HASH ON building_tag_data (tag)"""
  conn.cursor().execute(index_date_query)
  conn.cursor().execute(index_latlong_query)
  conn.cursor().execute(index_tag_query)
  conn.commit()


# pcd8 	oa21cd 	lsoa21cd 	msoa21cd 	ladcd 	LAD23CD 	LAD23NM
def create_postcode_to_areas(conn):
  drop = "DROP TABLE IF EXISTS postcode_area_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `postcode_area_data` (
            postcode VARCHAR(15) NOT NULL,
            oa21 VARCHAR(10) NOT NULL,
            lsoa21 VARCHAR(10) NOT NULL,
            msoa21 VARCHAR(10) NOT NULL,
            lad21 VARCHAR(10) NOT NULL,
            lad23 VARCHAR(10) NOT NULL,
            lad_name VARCHAR(40) NOT NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE postcode_area_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE postcode_area_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()

  
def create_postcode_to_area_index(conn):
  index_postcode = """CREATE INDEX idx_postcode_area_postcode USING HASH ON postcode_area_data (postcode)"""
  index_output_area = """CREATE INDEX idx_postcode_area_oa USING HASH ON postcode_area_data (oa21)"""
  index_lad = """CREATE INDEX idx_postcode_area_lad USING HASH ON postcode_area_data (lad23)"""
  conn.cursor().execute(index_postcode)
  conn.cursor().execute(index_output_area)
  conn.cursor().execute(index_lad)
  conn.commit()


def create_new_build_oa(conn):
  drop = "DROP TABLE IF EXISTS new_build_oa_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `new_build_oa_data` (
            postcode varchar(8) COLLATE utf8_bin NOT NULL,
            year int unsigned NOT NULL,
            property_type varchar(1) COLLATE utf8_bin NOT NULL,
            count int unsigned NOT NULL,
            oa21 VARCHAR(10) NOT NULL,
            lsoa21 VARCHAR(10) NOT NULL,
            msoa21 VARCHAR(10) NOT NULL,
            lad23 VARCHAR(10) NOT NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE new_build_oa_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE new_build_oa_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()


def create_electoral_data(conn):
  drop = "DROP TABLE IF EXISTS electoral_data"
  create_query = """
          CREATE TABLE IF NOT EXISTS `electoral_data` (
            ons_id varchar(10) COLLATE utf8_bin NOT NULL,
            year int unsigned NOT NULL,
            first_party varchar(5) COLLATE utf8_bin NOT NULL,
            electorate int unsigned NOT NULL,
            valid_votes int unsigned NOT NULL,
            con decimal(8,6) NULL,
            lab decimal(8,6) NULL,
            ld decimal(8,6) NULL,
            ruk decimal(8,6) NULL,
            green decimal(8,6) NULL,
            db_id bigint(20) unsigned NOT NULL
          ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1"""

  add_primary_key = "ALTER TABLE electoral_data ADD PRIMARY KEY (db_id)";
  auto_increment = "ALTER TABLE electoral_data MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT = 1";
  conn.cursor().execute(drop)
  conn.cursor().execute(create_query)
  conn.cursor().execute(add_primary_key)
  conn.cursor().execute(auto_increment)
  conn.commit()