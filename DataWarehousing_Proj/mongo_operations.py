import pandas as pd
from sodapy import Socrata
import pymongo
import time
from config import *

def fetch_mongo_data():
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

    # Start time for creating the MongoDB database
    start_db_creation = time.time()

    # Establishing connection with MongoDB and creating the database
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[MONGO_DATABASE]

    # End time for creating the MongoDB database
    end_db_creation = time.time()

    # Start time for creating collections
    start_collection_creation = time.time()

    cases_collection = db["cases"]
    patient_info_collection = db["patientinformation"]
    patient_health_status_collection = db["patienthealthstatus"]

    # End time for creating collections
    end_collection_creation = time.time()

    # Start time for inserting data
    start_insertion = time.time()

    # Inserting data into collections
    cases_collection.insert_many(results_df[['case_id','cdc_case_earliest_dt','cdc_report_dt','current_status','pos_spec_dt','onset_dt']].to_dict('records'))
    patient_info_collection.insert_many(results_df[['patient_id', 'case_id', 'sex', 'age_group', 'race_ethnicity_combined']].to_dict('records'))
    patient_health_status_collection.insert_many(results_df[['health_status_id', 'patient_id', 'hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn']].to_dict('records'))

    # End time for inserting data
    end_insertion = time.time()

    # Total time for the entire process
    end_time = time.time()

    # Capturing times instead of printing
    execution_time = {
        'db_creation': end_db_creation - start_db_creation,
        'collection_creation': end_collection_creation - start_collection_creation,
        'data_insertion': end_insertion - start_insertion,
        'total_execution': end_time - start_time
    }

    return execution_time
