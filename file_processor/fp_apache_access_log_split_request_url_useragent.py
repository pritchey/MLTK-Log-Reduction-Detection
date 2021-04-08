import apachelogs # Provides a nice parser for the Apache access log files.
import time # Imported so execution time can be captured.
import pyhash # pyhash used for access to FNV hash algorithm.
import numpy as np # Imported to access Numpy data types which underlay Pandas.
import pandas as pd # Using Pandas dataframes for performance and code clarity.
from pympler.asizeof import asizeof # Used to get more accurate memory utilization.
from sklearn.preprocessing import MinMaxScaler # Used to scale values so they fall between 0 and 1.

class simPlugin(object):
	state = ""

	# The following are used to capture features formed by splitting features contained in
	# the log file.  The matrices will eventually be appended to main matrix, the column name
	# list is dynamically built to match the number of split features extracted and will be
	# used to provide meaningful names when the matrix is typecast to a Pandas data frame.
	max_url_splits = 8
	max_user_agent_splits = 10
	dynamic_column_names=[]

	# Defines the Apache log file format we want to read.
	# See here:  https://apachelogs.readthedocs.io/en/stable/utils.html
	# And here:  http://httpd.apache.org/docs/current/mod/mod_log_config.html
	parse_format = apachelogs.COMBINED #  "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\""

	raw_data = []
	df = []

	url_splits = []
	user_agent_splits = []

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: fp_apache_access_log_split_request_url\n"
		text += "\t\tThis file parser processes Apache access log files.  After splitting the request into base\n"
		text += "\t\tcomponents (command (GET, PUT, ...), URL, and HTTP protocol version), the URL portion is\n"
		text += "\t\tsplit on the '/' character into additional features.\n"
		return text

	def build_Column_Names(self, stock_names):
		"""Dynamically build the list of column names for the Pandas DataFrames.  This must be done due to dynamic splitting
		of some values extracted from the log file, and to provide meaningful columns names in the CSV file if the user
		opts to save internal data structures for review.

		Args:
			stock_names (Array): Array containing the default, base (static) column names that dynamically generated names will
			be appended to.
		"""
		# Build our initial feature column names from the base set as extracted from
		# the log file which are not dynamically generated.
		self.dynamic_column_names = stock_names

		# To get the proper count of features, go ahead and typecast the Python matrix to
		# a Pandas data frame.  (We'll need this later for concatenation anyhow.)
		self.url_splits = pd.DataFrame(self.url_splits)
		for i in range(self.url_splits.shape[1]):
			self.dynamic_column_names.append("url_split_" + str(i))

		# Now do the same for the split user agent string.
		self.user_agent_splits = pd.DataFrame(self.user_agent_splits)
		for i in range(self.user_agent_splits.shape[1]):
			self.dynamic_column_names.append("user_agent_split_" + str(i))

	def split_string(self, split_on, the_string, max_splits):
		"""Perform deeper splitting on the URL using a specified character, but limiting the number of splits
		to a maximum.

		Args:
			split_on (String): Character to split on.
			the_string (String): String to be split.
			max_splits (Integer): Maximum number of splits to perform

		Returns:
			Array: Array containing the values obtained by splitting the string.
		"""
		# First, remove trailing split character if present so we don't end up with a blank
		# split result at the end.
		the_string = the_string.lstrip(split_on).rstrip(split_on)

		# Now split the string the specified maximum number of times with the remainder contained
		# in the final value if it exceeds the maximum number of splits.
		return the_string.split(split_on, max_splits)

	def url_splitter(self, url):
		"""Split a given URL into pieces based on the locations of '/'.

		Args:
			url (String): String containing the requested URL from the Apache log file.
		"""
		self.url_splits.append(self.split_string("/", url, self.max_url_splits))

	def user_agent_splitter(self, user_agent):
		"""
		user_agent: The user agent string from the request to be split apart.
		"""
		# Make sure we have a user agent - if we don't, set it to a default value.
		if type(user_agent) != str:
			user_agent = "No_User_Agent"
		self.user_agent_splits.append(self.split_string(";", user_agent.replace("(", ";").replace(")", ";"), self.max_user_agent_splits))

	def transform(self):
		"""Transform the data contained in the raw data Pandas DataFrame into something a machine learning algorithm can use.
		"""
		print("\t--Beginning:  Pandas dataframe transformation")

		# Capture start time.
		start_time = time.time()

		print("\nPre-transformation:\n")
		print(self.df.describe(include='all'))

		# Use FVN hash to transform string values into a numerical representation.
		print("\n\tBeginning:  Hashing and scaling string values.")
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
	
	def process_line(self, entry):
		"""Perform additional processing on a single Apache log line than apachelogs natively provides.

		Args:
			entry (Dictionary): apachelogs dictionary structure containing the parsed log line.
		"""
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
		temp = [entry.request_time.year, entry.request_time.month, entry.request_time.day, entry.request_time.hour,
		entry.request_time.minute, entry.request_time.second, entry.remote_host, entry.bytes_sent, category, 
		entry.headers_in["Referer"], command, parameters, version]

		# Split the URL into smaller components which will be appended later to the raw_data feature matrix.
		self.url_splitter(url)

		# Split the user agent into smaller components which will be appended later to the raw_data feature matrix.
		self.user_agent_splitter(entry.headers_in["User-Agent"])

		# Append the new row to the current raw data matrix.
		self.raw_data.append(temp)
		
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

		# Instantiate the apachelog parser object - this parses the log file automatically
		# for us.  No need to write our own code.
		parser = apachelogs.LogParser(self.parse_format)

		# Make sure we can open the log file - if we can't let the user know and exit immediately.
		try:
			fp = open(fileName)
		except Exception as e:
			print("\n\nERROR:  Unable to open log file -  %s\n\n" % str(e))
			exit()

		# Capture start time.
		start_time = time.time()
		print("\tBeginning to read file...")
		for entry in parser.parse_lines(fp):
			self.process_line(entry)
		print("\tFinished reading file...")

		print("\tFile Reading Time: %.4f seconds" % (time.time() - start_time))

		print("\tBeginning Creating Pandas dataframe...")

		# Before constructing the final data frame, build out the feature (column) names.  Part of
		# this is static, part of it is dynamic based on splitting features contained in the log file
		# which can vary.
		self.build_Column_Names(["year", "month", "day", "hour", "minute", "second", "remote_host", "bytes_sent", "status", "referer", "command", "parameters", "version"])

		# Take the raw data, contained in a native Python data structure and typecast to data frame.
		self.df = pd.DataFrame(self.raw_data)

		print("\t\tConcatenating dynamically split features to data frame...")
		self.df = pd.concat([self.df, self.url_splits, self.user_agent_splits], axis=1, ignore_index=True)

		# Apply column names - both the static "as taken from the log file" and dynamically generated from
		# splitting feature(s) into additional features.
		self.df.columns = self.dynamic_column_names

		# Now that the complete matrix containing raw data has been created, save it for later reference.
		self.raw_data = self.df.copy()
		print("\tFinished Creating Pandas dataframe...")

		print("\tSample Entry: \n----------\n%s\n---------\n" % self.df.iloc[0])

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
