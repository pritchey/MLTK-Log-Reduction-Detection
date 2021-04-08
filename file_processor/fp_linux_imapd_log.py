import time # Imported so execution time can be captured
import pyhash # pyhash used for access to FNV hash algorithm.
import numpy as np # Imported to access Numpy data types which underlay Pandas.
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame

class simPlugin(object):
	state = ""

	# Internal storage for data matrices built during processing.  (Raw data and
	# the final Pandas dataframe containing transformed data suitable for ML use.)
	raw_data = []
	df = []

	# Maximum number of times the log line should be split into features.
	maxsplit = 25

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: fp_linux_messages_log\n\t\tThis file parser processes Linux messages log files (/var/log/messages).\n"

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

		print("\tFinishd:  Hashing and scaling string values.")

		print("\n\nPost-transformation:")
		print(self.df.describe(include='all'))

		print("\t\tTransformation Time: %.4f seconds" % (time.time() - start_time))
		print("\t--Finished:  Pandas dataframe transformation")

	def process_file(self, fileName):
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
		max_columns = 0
		for entry in fp:
			# Remove double spaces (white space) that occur naturally in the log format for alignment purposes.
			# Remove double quote (") characters as well.
			deduplicated = entry.replace("  ", " ").replace("  ", " ").replace('"', "")
			extracted = deduplicated.split(" ", maxsplit=self.maxsplit)
			log_date = extracted[0] + " " + extracted[1] + " " + extracted[2]
			log_host = extracted[3]
			log_daemon = extracted[4].split("[")[0]
			row = [log_date, log_host, log_daemon]

			# Due to splitting the log line, the remainder past the daemon will vary.  Dynamically append.
			for z in extracted[5:]:
				row.append(z)
			if max_columns < len(row):
				max_columns = len(row)
			self.raw_data.append(row)

		# Dynamically build the column names.  The number of columns will vary based upon log message content splitting.
		column_names = ["date_time", "host", "daemon"]
		for count in range(max_columns - len(column_names)):
			column_names.append("message_" + str(count))

		print("\tSample Entry: %s" % self.raw_data[0])
		print("\tRaw data dimensions: %i x %i" % (len(self.raw_data[0]), len(self.raw_data)))
		print("\tRaw data memory: %iMB" % (asizeof(self.raw_data) / 1024 / 1024))
		print("\tFile Reading Time: %.4f seconds" % (time.time() - start_time))

		print("\tCreating Pandas dataframe...")
		self.df = DataFrame(self.raw_data, columns=column_names)

		# Remove the date/time feature.  Unfortunately we do it now because Python list does not support multi-dimensional slicing.
		print("\tPre-feature reduction data frame dimensions: %i x %i" % (self.df.shape[1], self.df.shape[0]))
		print("\tPre-feature reduction data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))
		self.df = self.df.iloc[:, 2:]

		print("\tPost-feature reduction data frame dimensions: %i x %i" % (self.df.shape[1], self.df.shape[0]))
		print("\tPost-feature reduction data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		# Transform the data contained in the Pandas dataframe into something usable by a
		# machine learning algorithm.
		self.transform()
		print("\tTransformed data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		print("--Finished:  File Processing")

		return self.raw_data, self.df