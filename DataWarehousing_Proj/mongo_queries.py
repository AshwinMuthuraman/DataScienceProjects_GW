from pymongo import MongoClient
import time
from config import *
import zlib
import base64

def fetch_mongo_queries():

    start_time = time.time()

    # Establish MongoDB connection
    mongo_client = MongoClient(MONGO_HOST, PORT)
    mongo_db = mongo_client[MONGO_DATABASE]
    mongo_collection_c = mongo_db['cases']

    #1. TIME SERIES indexing
    start_time_index = time.time()

    # Create an index on cdc_report_dt and onset_dt for faster sorting
    mongo_collection_c.create_index([("cdc_report_dt", 1), ("onset_dt", -1)])

    # Find patients with the longest recovery time
    result_mongo = list(mongo_collection_c.find(
        {"current_status": "Laboratory-confirmed case"},
        {"_id": 0, "case_id": 1, "cdc_report_dt": 1, "onset_dt": 1}
    ).sort([("cdc_report_dt", 1), ("onset_dt", -1)]))

    end_time_index = time.time()

    #print(list(result_mongo))

    #2.Data compression
    start_time_dc = time.time()
    # Fetch data from MongoDB
    data = list(mongo_collection_c.find())

    # Compress data using gzip and encode to base64
    compressed_data = base64.b64encode(zlib.compress(str(data).encode())).decode()

    # Insert compressed data back to MongoDB or use it as needed
    end_time_dc = time.time()


    #3.Text Search

    mongo_collection_phs = mongo_db['patienthealthstatus']

    start_time_ts = time.time()
    # Creating a text index on the field you want to search (medcond_yn)
    mongo_collection_phs.create_index([("medcond_yn", "text")])

    # Perform text search query
    search_keyword = "Missing"  # Replace with your search term
    result_mongo = list(mongo_collection_phs.find({"$text": {"$search": search_keyword}}))

    end_time_ts = time.time()


    #4.SUBQUERIES

    mongo_collection_pi = mongo_db['patientinformation']
    mongo_collection_phs = mongo_db['patienthealthstatus']
    mongo_collection_c = mongo_db['cases']

    # Measure execution time for subquery in MongoDB
    start_time_sq = time.time()

    # Subquery 1: Find distinct patient IDs with 'Yes' in medcond_yn field
    distinct_patients = mongo_collection_phs.distinct("patient_id", {"medcond_yn": "Yes"})

    # Subquery 2: Fetch patient information with associated cases for the filtered patients
    result_mongo = mongo_collection_pi.find(
        {"patient_id": {"$in": distinct_patients}},
        {
            "_id": 0,
            "patient_id": 1,
            "sex": 1,
            "age_group": 1,
            "race_ethnicity_combined": 1,
            "associated_cases": {
                "$map": {
                    "input": {
                        "$filter": {
                            "input": "$cases",
                            "as": "case",
                            "cond": {"$in": ["$$case.case_id", "$associated_cases"]}
                        }
                    },
                    "as": "filtered_case",
                    "in": "$$filtered_case.case_id"
                }
            }
        }
    )

    end_time_sq = time.time()

    #5.TIME SERIES CONTEXT , WINDOWS FUNCTION VS AGGREGATION

    start_time_wf = time.time()

    # Aggregation pipeline for MongoDB
    pipeline = [
        {
            "$group": {
                "_id": "$cdc_report_dt",
                "total_cases": {"$sum": 1}
            }
        },
        {
            "$sort": {"_id": 1}
        },
        {
            "$group": {
                "_id": None,
                "data": {"$push": {"date": "$_id", "total_cases": "$total_cases"}}
            }
        },
        {
            "$unwind": {
                "path": "$data",
                "includeArrayIndex": "data.index",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$group": {
                "_id": None,
                "data": {"$push": "$data"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "data": {
                    "$map": {
                        "input": "$data",
                        "as": "item",
                        "in": {
                            "date": "$$item.date",
                            "cumulative_cases": {
                                "$reduce": {
                                    "input": {"$slice": ["$data", {"$add": ["$$item.index", 1]}]},
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", "$$this.total_cases"]}
                                }
                            }
                        }
                    }
                }
            }
        }
    ]

    result_mongo = list(mongo_collection_c.aggregate(pipeline))
    end_time_wf = time.time()

    end_time = time.time()

    execution_time = {
        'Time Series Indexing Query': end_time_index - start_time_index ,
        'Data Compression Query': end_time_dc - start_time_dc  ,
        'Text Search Query': end_time_ts - start_time_ts,
        'Subqueries': end_time_sq - start_time_sq,
        'Time Series Aggregation': end_time_wf - start_time_wf ,
        'Total_execution': end_time - start_time 
    }

    return execution_time
    #mongo_client.close()