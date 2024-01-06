import mysql.connector
import time
from config import *
import zlib
import base64


def fetch_mysql_queries():

    start_time = time.time()

    # Establish MySQL connection
    mysql_conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    mysql_cursor = mysql_conn.cursor()

    #1. TIME-SERIES indexing 

    start_time_index = time.time()
    # Create an index on cdc_report_dt and onset_dt for faster sorting
    create_index_query = """
        CREATE INDEX idx_report_onset ON cases (cdc_report_dt ASC, onset_dt DESC)
    """
    mysql_cursor.execute(create_index_query)
    mysql_conn.commit()

    # Execute SQL query to find patients with the longest recovery time
    query = """
        SELECT 
            case_id,
            cdc_report_dt,
            onset_dt
        FROM 
            cases
        WHERE 
            current_status = 'Laboratory-confirmed case'
        ORDER BY 
            cdc_report_dt ASC, onset_dt DESC
    """

    mysql_cursor.execute(query)

    # Fetch the results
    result_mysql = mysql_cursor.fetchall()

    end_time_index = time.time()
    #print(list(result_mysql))
    print(mysql_cursor)

    #2. Data compression
    start_time_dc = time.time()

    # Execute SQL query to fetch data from MySQL
    query = "SELECT * FROM cases"
    mysql_cursor.execute(query)
    data = mysql_cursor.fetchall()

    # Compress data using gzip and encode to base64
    compressed_data = base64.b64encode(zlib.compress(str(data).encode())).decode()

    # Insert compressed data back to MySQL or use it as needed
    print(mysql_cursor)
    end_time_dc = time.time()


    #3.Text search

    # Check if the index exists before attempting to create it
    check_index_query = """
        SELECT COUNT(1) IndexExists
        FROM information_schema.statistics
        WHERE table_schema = 'cdc_cdc2'
        AND table_name = 'patienthealthstatus'
        AND index_name = 'medcond_index'
    """

    mysql_cursor.execute(check_index_query)
    index_exists = mysql_cursor.fetchone()[0]

    if not index_exists:
        create_index_query = """
            CREATE FULLTEXT INDEX medcond_index ON patienthealthstatus(medcond_yn)
        """
        mysql_cursor.execute(create_index_query)
        mysql_conn.commit()  # Commit the index creation

    # Perform full-text search query
    start_time_ts = time.time()

    search_keyword = "Missing"  # Replace with your search term
    sql_query = """
        SELECT * FROM patienthealthstatus 
        WHERE MATCH(medcond_yn) AGAINST (%s IN NATURAL LANGUAGE MODE)
    """
    mysql_cursor.execute(sql_query, (search_keyword,))

    end_time_ts = time.time()



    #4. SUBQUERIES

    # Measure execution time for subquery in MySQL
    start_time_sq = time.time()

    # Subquery 1: Find distinct patient IDs with 'Yes' in medcond_yn field
    mysql_cursor.execute(
        """
        SELECT DISTINCT patient_id 
        FROM patienthealthstatus 
        WHERE medcond_yn = 'Yes'
        """
    )
    distinct_patients = [patient[0] for patient in mysql_cursor.fetchall()]

    # Subquery 2: Fetch patient information with associated cases for the filtered patients
    mysql_cursor.execute(
        """
        SELECT 
            pi.patient_id, pi.sex, pi.age_group, pi.race_ethnicity_combined,
            GROUP_CONCAT(c.case_id SEPARATOR ', ') AS associated_cases
        FROM 
            patientinformation pi
        JOIN 
            cases c ON pi.case_id = c.case_id
        WHERE 
            pi.patient_id IN ({})
        GROUP BY 
            pi.patient_id
        """.format(','.join(['%s'] * len(distinct_patients)))
        , distinct_patients
    )
    result_mysql = mysql_cursor.fetchall()

    end_time_sq = time.time()


    #5.TIME SERIES CONTEXT , WINDOWS FUNCTION VS AGGREGATION

    start_time_wf = time.time()

    # Execute SQL query
    query = """
        SELECT 
            cdc_report_dt,
            SUM(total_cases) OVER (ORDER BY cdc_report_dt) AS cumulative_cases
        FROM (
            SELECT cdc_report_dt, COUNT(*) AS total_cases 
            FROM cases 
            GROUP BY cdc_report_dt
        ) AS subquery
    """

    
    mysql_cursor.execute(query)

    # Fetch the results
    result_wf = mysql_cursor.fetchall()

    end_time_wf = time.time()

    execution_time_wf = end_time_wf - start_time_wf
    print(f"MySQL Execution Time: {execution_time_wf} seconds")

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