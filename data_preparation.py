import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='data_processing.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def process_yearly_data(year, base_path):
    quarterly_data = []

    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        try:
            file_path = os.path.join(base_path, f'nyc-park-crime-stats-{quarter}-{year}.xlsx')
            
            # Adjust skiprows based on specific years and quarters
            if (year == 2015 and quarter in ['Q1', 'Q2', 'Q3']) or \
               (year == 2018 and quarter == 'Q2') or \
               (year == 2021 and quarter == 'Q1'):
                skiprows = 4
            else:
                skiprows = 3
            
            # Load the data
            data = pd.read_excel(file_path, skiprows=skiprows)
            
            # Normalize column names to uppercase
            data.columns = data.columns.str.upper()
            
            # Filtering out rows where PARK equals "Total" (after normalizing to uppercase)
            data = data[data['PARK'] != 'Total']
            
            # Add quarter and year information
            data['QUARTER'] = quarter
            data['YEAR'] = year
            
            quarterly_data.append(data)
        
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    if not quarterly_data:
        logging.error(f"No data processed for year {year}.")
        return pd.DataFrame()

    year_data = pd.concat(quarterly_data, ignore_index=True)
    
    # Fill missing values and standardize names (further cleaning if necessary)
    year_data.fillna(0, inplace=True)
    year_data['PARK'] = year_data['PARK'].str.title().str.strip()
    year_data['BOROUGH'] = year_data['BOROUGH'].str.upper().str.strip()
    
    return year_data


# Setting up the paths
project_root = os.getcwd()
base_path = os.path.join(project_root, 'data', 'raw')
processed_path = os.path.join(project_root, 'data', 'processed')

all_years_data = []

# Process data for each year
for year in range(2015, 2024):
    logging.info(f"Processing data for year {year}.")
    yearly_data = process_yearly_data(year, base_path)
    
    if not yearly_data.empty:
        all_years_data.append(yearly_data)
    else:
        logging.info(f"No data added for year {year}.")

# Merge and save the processed data
if all_years_data:
    final_data = pd.concat(all_years_data, ignore_index=True)
    
    # Normalize column names for the final DataFrame if not already done
    final_data.columns = final_data.columns.str.upper()
    
    # Convert 'YEAR' and 'QUARTER' to categorical types for analysis
    final_data['YEAR'] = pd.Categorical(final_data['YEAR'])
    final_data['QUARTER'] = pd.Categorical(final_data['QUARTER'], categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)

    final_data = final_data[final_data['PARK'].str.upper() != 'TOTAL']
    
    # Save the consolidated data
    final_file_path = os.path.join(processed_path, 'nyc_park_crime_data_2015_to_2023.csv')
    final_data.to_csv(final_file_path, index=False)
    logging.info("Data processing completed successfully and saved.")
else:
    logging.error("No data processed.")
