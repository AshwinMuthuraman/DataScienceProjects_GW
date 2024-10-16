# Integrating-Apache-Hadoop-and-Apache-Spark-for-ComprehensiveData-Analysis
A data analysis project involving customer segmentation using PySpark and Hadoop. This project is configured to run Spark in standalone mode.

## Project Structure

```
Big_Data_Project/
│
├── data/
│ ├── fname=0.csv
│ ├── fname=1.csv
│ ├── fname=2.csv
│ ├── fname=3.csv
│ ├── temp_H.txt
│ ├── missing_values_plot.png
│ ├── Frequency_log_plot.png
│ ├── Frequency_plot.png
│ ├── Monetary_histogram.png
│ ├── Monetary_log_histogram.png
│ ├── Monetary_log_plot.png
│ ├── Monetary_plot.png
│ ├── Recency_Boxcox_histogram.png
│ ├── Recency_Boxcox_plot.png
│ ├── Recency_histogram.png
│ └── Recency_plot.png
│
├── scripts/
│ ├── Big_Data_Project.ipynb
│ └── Utility_Folder/
│ ├── utility.py
│ └── init.py
│
└── README.md

```


## Setup

1. Clone the Repository:

   ```sh
   git clone <repository_url>
   cd Big_Data_Project

2. Install Dependencies:

Ensure you have Python 3.7+ and pip installed.

```
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

To explore the analysis interactively, run the Jupyter notebook.

```
jupyter notebook scripts/Big_Data_Project.ipynb
```

4. Convert Notebook to Python Script:

After completing the notebook, save it as a Python script:

In the Jupyter notebook interface, go to File -> Download as -> Python (.py)
Save the file as Big_Data_Project.py in the scripts/ directory.

5. Run the Python Script as a Spark Job:

To execute the analysis as a Spark job, run the following command:

```
spark-submit --master "local[*]" scripts/Big_Data_Project.py hdfs://localhost:9000/project/full_data.csv
```

## Utility Functions

The utility functions are located in scripts/Utility_Folder/utility.py. They include:
```
   (i)elbow
   (ii)clust_plot
   (iii)plot_data
   (iv)outlier
```
These functions are used for visualization and data analysis within the project.

## Data Files

Data files used in the analysis are located in the data/ directory. Ensure the paths in the scripts point to the correct locations of these files.

## Visualization Outputs

Visualization outputs (e.g., plots) are saved in the data/ directory. Ensure this directory exists and is writable before running the scripts.

## Notes
```
   (i)Ensure Hadoop and Spark are correctly set up and configured on your system.
   (ii)Adjust any file paths in the scripts if necessary to match your local environment setup.
   (iii)Monitoring Spark Job: You can monitor the progress and performance of the Spark job using the Spark UI. By default, the Spark UI runs on port 4040. 
       Open web browser and go to http://localhost:4040 to access the Spark UI. The UI provides detailed information about the running Spark job, including 
       stages, tasks, storage, environment, and executors. This helps in monitoring the progress and performance of the job.
```

