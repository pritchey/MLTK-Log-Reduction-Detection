# README #

The code contained in this repository is a machine learning toolkit for performing log reduction and the detection of log lines indicating potentially malicious behavior.  The code was created as part of a research project undertaken while pursuing a Master of Science in CyberSecurity at Villanova University.

# Requirements
The Python scripts contained in this source code repository were written specifically with Python version 3.x.  Natively provided libraries are used when possible, however there are several highly specialized libraries providing some functionality that are required:

```
#Provides simple API for extracting data from Apache access log files.
pip3 install apachelogs

#Provides more accurate memory utilization statistics for Python objects.
pip3 install pympler

#Provides access to optimized methods and data structures for working with large amounts of data
pip3 install numpy

#Provides large scale data manipulation and handling of data sets capability.
pip3 install pandas

#Provides numerous machine learning algorithms and data manipulation capabilities
pip3 install sklearn

# Provides access to FVN hashing algorithm which is no longer natively provided by Python's hashlib.
pip3 install pyhash

# Provides an interactive graphing capability.
pip3 install plotly

# Provided alternative graphing capability but was not as performant as plotly.
pip3 install seaborn
```

# Basic Command Line Usage

Basic help information is available using the sample command below.  Information for all available command line flags will be printed, as well as all available plugins and their intended purpose.

```
python3 log_anomaly_identifier.py -h
```

Sample command to read and process an Apache access log file using TSVD chained with k-means for reduction/detection::

```
python3 log_anomaly_identifier.py -f fp_apache_access_log -l apache-access.log -m ml_scikit_tsvd -n ml_scikit_kmeans 
```
