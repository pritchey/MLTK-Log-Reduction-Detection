import time # Imported so execution time can be captured
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame

class simPlugin(object):
	state = ""

	raw_data = []
	df = []

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: fp_apache_error_log\n\t\tThis file parser processes Apache error log files.\n"

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
			fp = open(fileName, "r")
		except Exception as e:
			print("\n\nERROR:  Unable to open log file -  %s\n\n" % str(e))
			exit()

		# Capture start time.
		start_time = time.time()
		for entry in fp:
			extracted = entry.split("] ")
			log_date = extracted[0][1:]
			log_client = (extracted[3][8:]).split(":")[0]
			if ", referer: " not in extracted[4]:
				log_error = extracted[4].rstrip()
				log_referer = "NONE"
			else:
				log_error, log_referer = (extracted[4]).split(", referer: ")

			self.raw_data.append([log_date, log_client, log_error, log_referer.rstrip()])
		print("\tSample Entry: %s" % self.raw_data[0])
		print("\tRaw data dimensions: %i x %i" % (len(self.raw_data[0]), len(self.raw_data)))
		print("\tRaw data memory: %iMB" % (asizeof(self.raw_data) / 1024 / 1024))
		print("\tFile Reading Time: %.4f seconds" % (time.time() - start_time))

		print("\tCreating Pandas dataframe...")
		self.df = DataFrame(self.raw_data, columns=["date_time", "source_ip", "message", "request"])

		print("--Finished:  File Processing")

		return self.raw_data, self.df