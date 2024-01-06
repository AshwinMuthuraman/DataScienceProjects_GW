from database_operations import fetch_cdc_data_and_store
from mongo_operations import fetch_mongo_data
from mongo_queries import fetch_mongo_queries
from mysql_queries import fetch_mysql_queries
from config import *

def main():
    # MySQL operations
    execution_times = fetch_cdc_data_and_store()
    print("MySQL Operations:")
    for key, value in execution_times.items():
        print(f"{key}: {value} seconds")
    
    # MySQL Queries
    mysql_query_time = fetch_mysql_queries()
    print("MySQL Queries:")
    for key, value in mysql_query_time.items():
        print(f"{key}: {value} seconds")

    # MongoDB operations
    mongo_execution_time = fetch_mongo_data()
    print("\nMongoDB Operations:")
    for key, value in mongo_execution_time.items():
        print(f"{key}: {value} seconds")

    # MongoDB Queries
    mongo_query_time = fetch_mongo_queries()
    print("Mongo Queries:")
    for key, value in mongo_query_time.items():
        print(f"{key}: {value} seconds")


if __name__ == "__main__":
    main()
