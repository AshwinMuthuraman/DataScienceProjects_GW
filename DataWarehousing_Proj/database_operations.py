import pandas as pd
from sodapy import Socrata
from sqlalchemy import create_engine
import mysql.connector
import time
from config import *

def fetch_cdc_data_and_store():
    start_time = time.time()

    # Connecting to Socrata and fetching data
    client = Socrata("data.cdc.gov", None)
    results = client.get("vbim-akqf", limit=500000)
    results_df = pd.DataFrame.from_records(results)

    # Adding IDs and replacing NaN values
    results_df['case_id'] = results_df.index + 1
    results_df['patient_id'] = results_df.index.map(lambda x: f"0{x + 1:02d}")
    results_df['health_status_id'] = results_df.index.map(lambda x: f"00{x + 1:03d}")
    results_df.replace({pd.NA: None, 'NaN': None}, inplace=True)

    # MySQL connection for creating the database
    db_connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD
    )

    # Creating a cursor object
    db_cursor = db_connection.cursor()

    # Start time for creating the database
    start_db_creation = time.time()

    # Create the database 'cdc_data13'
    #db_cursor.execute("CREATE DATABASE {MYSQL_DATABASE}")

    create_db_query = f"CREATE DATABASE {MYSQL_DATABASE}"
    db_cursor.execute(create_db_query)

    # Commit and close the cursor and database connection
    db_connection.commit()
    db_cursor.close()
    db_connection.close()

    # End time for creating the database
    end_db_creation = time.time()

    # Reconnect to the newly created database
    db_connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

    # Creating a cursor object for the new connection
    db_cursor = db_connection.cursor()

    # Start time for creating tables
    start_table_creation = time.time()

    # Creating tables using mysql.connector
    db_cursor.execute("""
        CREATE TABLE cases (
            case_id VARCHAR(100) PRIMARY KEY,
            cdc_case_earliest_dt DATETIME,
            cdc_report_dt DATETIME,
            current_status VARCHAR(100),
            pos_spec_dt DATETIME,
            onset_dt DATETIME
        )
    """)

    db_cursor.execute("""
        CREATE TABLE patientinformation (
            patient_id VARCHAR(100) PRIMARY KEY,
            case_id VARCHAR(100),
            sex VARCHAR(100),
            age_group VARCHAR(100),
            race_ethnicity_combined VARCHAR(100),
            FOREIGN KEY (case_id) REFERENCES Cases(case_id)
        )
    """)

    db_cursor.execute("""
        CREATE TABLE patienthealthstatus (
            health_status_id VARCHAR(100) PRIMARY KEY,
            patient_id VARCHAR(100),
            hosp_yn VARCHAR(100),
            icu_yn VARCHAR(100),
            death_yn VARCHAR(100),
            medcond_yn VARCHAR(100),
            FOREIGN KEY (patient_id) REFERENCES PatientInformation(patient_id)
        )
    """)

    # Commit and close the cursor and database connection
    db_connection.commit()
    db_cursor.close()
    db_connection.close()

    # End time for creating tables
    end_table_creation = time.time()

    # Create a new engine for the 'cdc_data13' database using sqlalchemy
 
    x = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    engine = create_engine(x)

    # Start time for data insertion
    start_insertion = time.time()

    # Inserting data into the tables using sqlalchemy
    results_df[['case_id','cdc_case_earliest_dt','cdc_report_dt','current_status','pos_spec_dt','onset_dt']].to_sql(
        name='cases', con=engine, if_exists='append', index=False)
    results_df[['patient_id', 'case_id', 'sex', 'age_group', 'race_ethnicity_combined']].to_sql(
        name='patientinformation', con=engine, if_exists='append', index=False)
    results_df[['health_status_id', 'patient_id', 'hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn']].to_sql(
        name='patienthealthstatus', con=engine, if_exists='append', index=False)

    # End time for data insertion
    end_insertion = time.time()

    # Total time for the entire process
    end_time = time.time()

    # Capturing times instead of printing
    execution_times = {
        'db_creation': end_db_creation - start_db_creation,
        'table_creation': end_table_creation - start_table_creation,
        'data_insertion': end_insertion - start_insertion,
        'total_execution': end_time - start_time
    }

    return execution_times
