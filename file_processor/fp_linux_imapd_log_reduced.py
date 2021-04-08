# process login, badlogin, and failed lines from imapd log
# extract what (login=0, badlogin=1, failed=2), IP (32-bit int), and userid (or "unknown")
#
# if imapd_feature_counts plugin option is specified then entries with the same IP and userid
# are combined into single entries with separate login, badlogin, and failed counts.

import time # Imported so execution time can be captured
import pyhash # pyhash used for access to FNV hash algorithm.
import numpy as np # Imported to access Numpy data types which underlay Pandas.
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame

import ipaddress # to convert IP to 32-bit int
import re # regex

IP_regex = re.compile(r"^\[\d+\.\d+\.\d+\.\d+\]$") # e.g. [192.168.0.1]

class simPlugin(object):
	state = ""

	# Internal storage for data matrices built during processing.  (Raw data and
	# the final Pandas dataframe containing transformed data suitable for ML use.)
	raw_data = []
	df = []

	# Maximum number of times the log line should be split into features.
	maxsplit = 15

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: fp_linux_imapd_log_reduced\n\t\tThis file parser processes Linux imapd log files which reduces the data by only extracting essential info from the impad log file.\n"

	def transform(self):
		"""Transform the data contained in the raw data Pandas DataFrame into something a machine learning algorithm can use.
		"""
		print("\t--Beginning:  Pandas dataframe transformation")

		# Capture start time.
		start_time = time.time()

		print("\nPre-transformation:\n")
		print(self.df.describe(include='all'))

		# Use FVN hash to transform string values into a numerical representation.
		print("\tBeginning:  Hashing and scaling string values.")
		hash_alg = pyhash.fnv1_64()

		# Examine the Pandas data frame column by column and hash/scale only
		# columns detected as strings.
		for i in self.df.columns:
			if self.df.dtypes[i] == np.object:
				print("\t\tHashing and scaling column: %s" % i)
				self.df[i] = self.df[i].map(lambda a: (hash_alg(str(a).encode('utf-8'))) / 2**64)

		print("\tFinished:  Hashing and scaling string values.")

		print("\n\nPost-transformation:")
		print(self.df.describe(include='all'))

		print("\t\tTransformation Time: %.4f seconds" % (time.time() - start_time))
		print("\t--Finished:  Pandas dataframe transformation")

	# update after fixing log_anomaly_identifier.py to pass plugin_options argument
	# def process_file(self, fileName, plugin_options):
	def process_file(self, fileName):

		# temporary:
		plugin_options = {} # { 'imapd_feature_counts' : 'true' }

		"""Read the specified file and create a raw data Pandas DataFrame, and a DataFrame with the transformed data ready for use
		in a machine learning algorithm.

		Args:
			fileName (String): Path and name of the file to be processed.

		Returns:
			Pandas DataFrames: A DataFrame containing the raw data as read from the file and parsed.  Another DataFrame containing the
			transformed data ready for use in a machine learning algorithm. 
		"""
		print("\n")
		print("--Beginning:  File Processing")
		print("\tFile being processed: %s" % fileName)

		try:
			fp = open(fileName, mode="r", errors="replace")
		except Exception as e:
			print("\n\nERROR:  Unable to open log file -  %s\n\n" % str(e))
			exit()

		# Capture start time.
		start_time = time.time()
		for entry in fp:
			extracted = entry.split(maxsplit=self.maxsplit)
#
# formats:
# login: IP user ...
# login: hostname IP user ...
# badlogin: IP plaintext user ...
# badlogin: hostname IP plaintext user ...
# badlogin: IP ... (no user info)
# badlogin: hostname IP ... (no user info)
# failed: IP (no user info)
# failed: hostname IP (no user info)
#
# examples:
# 5      6             7      8
# login: [172.58.7.94] eileen plaintext+TLS ...
# 5      6                   7                8
# login: docker1.morroni.com [198.178.251.20] mor0117 ...
# 5         6                7         8
# badlogin: [174.77.111.197] plaintext sales@intekusa.com SASL(-13): user not found: checkpass failed
# 5         6                                 7                 8         9
# badlogin: 191-101-130-130-hostedby.bcr.host [191.101.130.130] plaintext don@misty.com invalid user
# 5         6                7     8
# badlogin: [183.89.243.161] plain [SASL(-13): ...
# 5         6              7               8
# badlogin: static.vnpt.vn [14.161.27.203] plain [SASL...
#     x       x+              		         x+2
# ... failed: c-76-116-49-54.hsd1.nj.comcast.net [76.116.49.54]
#
			x = 5; login = 0; badlogin = 0; failed = 0

			if extracted[x] == "login:":
				what = 0; login = 1;
			elif extracted[x] == "badlogin:":
				what = 1; badlogin = 1;
			else:
				x = 0
				for i in range(6,len(extracted)):
					if extracted[i] == "failed:": x = i; break
				if x:
					what = 2; failed = 1;
				else:
					continue

# map this IP to 0:
# login: localhost [::1] mailadm ...

			ip_index = 0
			if re.search(IP_regex,extracted[x+1]):
				ip_index = x+1
			elif re.search(IP_regex,extracted[x+2]):
				ip_index = x+2

			if ip_index:
				IP = int(ipaddress.IPv4Address(extracted[ip_index][1:-1]))
			else:
				IP = 0; ip_index = x+2

			if what == 0:
				userid = extracted[ip_index+1]
			elif what == 1 and extracted[ip_index+1] == "plaintext":
				userid = extracted[ip_index+2]
			else:
				userid = "unknown"

			if "imapd_feature_counts" in plugin_options:
				self.raw_data.append([login, badlogin, failed, IP, userid])
			else:
				self.raw_data.append([what, IP, userid])

		# print( self.raw_data) # debug
		print("\tSample Entry: %s" % self.raw_data[0])

		if "imapd_feature_counts" in plugin_options:
			print("\tPreliminary raw data dimensions: %i x %i" % (len(self.raw_data[0]), len(self.raw_data)))
			print("\tSorting and combining...")
			r = np.array(self.raw_data)
			s = np.lexsort((r[:,4],r[:,3])) # sort by IP, then userid
			r = []; j = -1 # r[j] will be current entry
			for i in s:
				IP = self.raw_data[i][3]
				userid = self.raw_data[i][4]
				if len(r) == 0 or IP != prev_IP or userid != prev_userid:
					prev_IP = IP
					prev_userid = userid
					r.append(self.raw_data[i])
					j += 1
				else:
					for k in range(0,3): r[j][k] += self.raw_data[i][k]
			self.raw_data = r;
			column_names = ["login", "badlogin", "failed", "IP", "userid"]
		else:
			column_names = ["what", "IP", "userid"]

		print("\tRaw data dimensions: %i x %i" % (len(self.raw_data[0]), len(self.raw_data)))
		print("\tRaw data memory: %iMB" % (asizeof(self.raw_data) / 1024 / 1024))
		print("\tFile Reading Time: %.4f seconds" % (time.time() - start_time))

		print("\tCreating Pandas dataframe...")
		self.df = DataFrame(self.raw_data, columns=column_names)

		print("\tData frame dimensions: %i x %i" % (self.df.shape[1], self.df.shape[0]))
		print("\tData frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		# Transform the data contained in the Pandas dataframe into something usable by a
		# machine learning algorithm.
		self.transform()
		print("\tTransformed data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		print("--Finished:  File Processing")

		return self.raw_data, self.df
