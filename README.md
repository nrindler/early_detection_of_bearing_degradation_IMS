# Early Detection of Bearing Degradation
Early detection of bearing degradation using IMS vibration signal data and time series segmentation methods

This is my capstone project as part of the **Spring 2024 Applied Analytics Practicum (MGT 6748)** at [**Georgia Tech Online Master’s in Analytics**](https://pe.gatech.edu/degrees/analytics).  The project sponsor was Sandia National Labs, who tasked students with building a classifier on signal data to identify "healthy" vs. "faulty" signal.

**Author**: Nadav Rindler (nrindler3)

### Abstract
Prognostics is an engineering discipline that applies data science and machine learning techniques to predict when part or machine maintenance is needed before fault or failure occurs.  These techniques can be applied to bearings, a critical machine component with wide industrial applications.  Bearing vibration signal can be measured using accelerometer sensors to determine states of deterioration and failure.  Until now, most research has focused on labeling bearing status as functional or failed, a binary classification problem.  Using multi-class classification techniques, it may be possible to detect early stages of bearing fault before failure occurs.  

First, I define three bearing life stages – healthy, faulty, and failed – and identify two time series measures from the literature that are predictive of failure. I then discuss an attempt to apply a semi-supervised learning approach developed by [Juodelyte, et. al.](https://arxiv.org/abs/2203.03259) to new bearing data from three run-to-failure experiments published by the Center for Intelligent Maintenance Systems (IMS) in 2007 and accessed via the open-source [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).  Finally, I develop a new unsupervised time series segmentation approach to detect phase change from the healthy to faulty states, allowing early detection before failure occurs.  This approach detects meaningful change in the bearing’s signal, on average XX% of the bearing’s lifetime in advance of actual failure and with sensitivity of XX% and specificity of XX% on experimental data from 12 bearings.  This approach is simpler, more reliable, and more accurate. This analysis shows that early detection of bearing degradation is possible and has implications for improved predictive maintenance.

### Full Text
See [Rindler - SP24 Practicum Final Report.pdf]().

### How to navigate this project
**Data**  
Bearing data are from three run-to-failure experiments published by the Center for Intelligent Maintenance Systems (IMS) in 2007 and are accessible from the open-source [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).  Sample intermediate datasets, and the folder structure used in the analysis, are provided to facilitate replication.

**Notebooks**
1. **Read_and_Sample_Data** - creates functions to ingest and down-sample data from the three run-to-failure bearing experiments. Functions are saved to "Read_and_Sample_Data.py" to be imported in subsequent notebooks.
2. **Data_Transforms** - creates functions to extract frequency-domain features via Fast Fourier Transform and calculate time-domain features via 22 measures of the signal data (absolute mean, RMS, kurtosis, etc.). Functions are saved to "Data_Transforms.py" to be imported in subsequent notebooks.
3. **Extract_Freq_and_Time_Series_Features** - reads data from the three experiments using functions from "Read_and_Sample_Data.py", extracts frequency- and time-domain features using functions from "Data_Transforms.py", and saves down the resulting frequency and time series data to CSV files, outputting two files per each of the 12 bearings in the IMS dataset.
4. **Exploratory Data Analysis** ("EDA-Test1.ipynb", "EDA-Test2.ipynb", and "EDA-Test3.ipynb") - show exploratory data analysis on each of the three experiments (each experiment included 4 bearings). The EDA evaluates various time series measures for predictive value in creating manual labels to classify bearings into healthy, faulty, and failed states.
5. **AutoEncoder** -  
