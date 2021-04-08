import apachelogs # Provides a nice parser for the Apache access log files.
import time # Imported so execution time can be captured.
import pyhash # pyhash used for access to FNV hash algorithm.
import numpy as np # Imported to access Numpy data types which underlay Pandas.
from pympler.asizeof import asizeof # Used to get more accurate memory utilization.
from pandas import DataFrame # Using Pandas dataframes for performance and code clarity.
from sklearn.preprocessing import MinMaxScaler # Used to scale values so they fall between 0 and 1.

class simPlugin(object):
	state = ""
	# Defines the Apache log file format we want to read.
	# See here:  https://apachelogs.readthedocs.io/en/stable/utils.html
	# And here:  http://httpd.apache.org/docs/current/mod/mod_log_config.html
	parse_format = apachelogs.COMBINED #  "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\""

	raw_data = []
	df = []

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: fp_apache_access_log_split_request\n"
		text += "\t\tThis file parser processes Apache access log files.  A simple split is performed on the request\n"
		text += "\t\tbreaking it into the command (GET, PUT, ...), URL, and HTTP protocol version.\n"
		return text

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

		parser = apachelogs.LogParser(self.parse_format)

		try:
			fp = open(fileName)
		except Exception as e:
			print("\n\nERROR:  Unable to open log file -  %s\n\n" % str(e))
			exit()

		# Capture start time.
		start_time = time.time()
		print("\tBeginning to read file...")
		for entry in parser.parse_lines(fp):
			# Split the request, taking care as there are three different structures for the request in the current
			# data set.
			command = (str(entry.request_line).split(" "))[0].strip()
			if command == "OPTIONS":
				url = "None"
				version = (str(entry.request_line).split(" "))[2]
				parameters = "None"
			elif command == "None":
				url = "None"
				version = "None"
				parameters = "None"
			else:
				temp = (str(entry.request_line)).split(" ")[1]
				version = (str(entry.request_line)).split(" ")[2]
				parameters = "None"
				# Check to see if there are parameters, if so we need to perform another split...
				if "?" in temp:
					# There are requests where a parameter contains a URL and parameter itself.  This
					# technique maintains the integrity of the original request/parameters.
					url = temp[:temp.find("?")]
					parameters = temp[temp.find("?") + 1:]
				else:
					# No parameters were included in the request.
					url = temp

			# Analyze the status code and translate to a numeric version used as a
			# new feature in the dataset.
			# https://ci.apache.org/projects/httpd/trunk/doxygen/group__HTTP__Status.html
			category = 0.0
			if entry.final_status >= 100 and entry.final_status < 200:
				category = 0.1 # Informational
			elif entry.final_status >= 200 and entry.final_status < 300:
				category = 0.2 # Success
			elif entry.final_status >= 300 and entry.final_status < 400:
				category = 0.3 # Redirection
			elif entry.final_status >= 400 and entry.final_status < 500:
				category = 0.4 # Client Error
			elif entry.final_status >= 500 and entry.final_status < 600:
				category = 0.5 # Server Error

			# Extract the raw values out of the parsed Apache log file line.
			# See here for documentation on extractable values:  https://apachelogs.readthedocs.io/en/stable/directives.html
			self.raw_data.append([entry.request_time.year, entry.request_time.month, entry.request_time.day, entry.request_time.hour,
			entry.request_time.minute, entry.request_time.second, entry.remote_host, entry.bytes_sent, category, 
			entry.headers_in["Referer"], entry.headers_in["User-Agent"], command, url, parameters, version]
			)
			
		print("\tFinished reading file...")
		print("\tSample Entry: %s" % self.raw_data[0])
		print("\tRaw data dimensions: %i x %i" % (len(self.raw_data[0]), len(self.raw_data)))
		print("\tRaw data memory: %iMB" % (asizeof(self.raw_data) / 1024 / 1024))
		print("\tFile Reading Time: %.4f seconds" % (time.time() - start_time))

		print("\tCreating Pandas dataframe...")
		self.df = DataFrame(self.raw_data, columns=["year", "month", "day", "hour", "minute", "second", "remote_host", "command", "url", "parameters", "version", "bytes_sent", "status", "referer", "user-agent"])

		# Remove the date/time features.  Unfortunately we do it now because Python list does not support multi-dimensional slicing.
		print("\tPre-feature reduction data frame dimensions: %i x %i" % (self.df.shape[1], self.df.shape[0]))
		print("\tPre-feature reduction data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))
		self.df = self.df.iloc[:, 6:]

		print("\tPost-feature reduction data frame dimensions: %i x %i" % (self.df.shape[1], self.df.shape[0]))
		print("\tPost-feature reduction data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		# Transform the data contained in the Pandas dataframe into something usable by a
		# machine learning algorithm.
		self.transform()
		print("\tTransformed data frame memory: %iMB" % (asizeof(self.df) / 1024 / 1024))

		return self.raw_data, self.df