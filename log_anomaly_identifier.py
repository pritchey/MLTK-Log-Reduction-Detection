import glob # Imported for file directory listing capabilities.
import sys # Imported for access to command line parameters
import os # Imported for file path manipulation capabilities.
import argparse # Imported for handling command line arguments.
import time # Imported so execution time can be captured
import pandas # Imported to access built in ability to easily save data frames to files.
import numpy as np # Imported to access numpy data types.
from pathlib import Path # Imported to facilitate autocreation of path to save data into.

# Dictionaries to store the various plugins
file_processor = {}
machine_learning = {}
output = {}
anomaly_reduction = {}

plugin_parameters = {}

label_data = pandas.DataFrame([])
ml_results = {}
ml_results2 = {}
anomaly_reduction_results = {}

# Integers used to store execution time for various steps performed as
# part of normal execution.
file_processor_time = 0
machine_learning_time = 0
machine_learning_time2 = 0
save_time = 0
label_time = 0
output_time = 0
anomaly_reduction_time = 0

def load_plugins(directory, whichDictionary):
	"""Load available plugins into separate dictionaries based upon their plugin type.

	Args:
		directory (String): Path/directory containing the plugins to load for a particular type.
		whichDictionary (Dictionary): Empty dictionary to load the plugins into.
	"""
	myPlugIns = sorted(glob.glob(directory + "/*.py"))
	for file in myPlugIns:
		# Extract just the first part of the .py file name.
		name = file.split("/")[1].split(".")[0]
		print ("Loading %s plugin: %s" % (directory, name))

		# Dynamically set the PYTHONPATH so the user doesn't have to. It assumes
		# the plugins are contained in subdirectories where the main file lives.
		path = os.path.dirname(sys.argv[0])
		if len(path) == 0:
			path = "."
		sys.path.append(path + "/" + directory)

		# Import the plugin module temporarily long enough to instantiate an object
		# which is stored in a globally accessible dictionary.
		tempModule = __import__(name)
		whichDictionary[name] = tempModule.simPlugin()

def available_plugins():
	"""When the user uses the help command line option, this builds a formatted text string describing
	all of the different plugins and what they do.

	Returns:
		String: Formatted string containing all the help output from each plugin.
	"""
	text = "Available 'file processing' plug-ins:\n"
	for i in file_processor:
		text += file_processor[i].print_help() + "\n"

	text += "\n\nAvailable 'machine learning' plug-ins:\n"
	for i in machine_learning:
		text += machine_learning[i].print_help() + "\n"

	text += "\n\nAvailable 'output' plug-ins:\n"
	for i in output:
		text += output[i].print_help() + "\n"

	text += "\n\nAvailable 'anomaly' plug-ins:\n"
	for i in anomaly_reduction:
		text += anomaly_reduction[i].print_help() + "\n"
	return text

def save_data(save_path, save_data):
	"""Loops through the supplied dictionary of internal data and saves each to a CSV file.

	Args:
		save_path (String): Path where the data should be saved.
		save_data (Dictionary): Dictionary containing all the Pandas DataFrames and dictionaries of internal data to be saved to disk. 
	"""
	print("--Beginning:  Saving Machine Learning Data")
	print("\tSaving data to path: %s" % save_path)
	Path(save_path).mkdir(parents=True, exist_ok=True)
	for i in save_data:
		# Ignore the "graph" entry if present - it's used to indicate which matrix
		# is used for graphing from the results returned by an ML plugin.
		if i != "graph":
			print("\tSaving Pandas data frame: %s" % i)
			pandas.DataFrame(save_data[i]).to_csv(save_path + "/" + i + ".csv")

	print("--Finished:  Saving Machine Learning Data")

def get_labels(label_data_file):
	"""Extract label data from the file specified by the user on the command line.  The assumed
	format for the label file is a CSV file where several different patterns can trigger a "1"
	label instead of a "0" label:

	<string>,0
	<string>,<string>
	0,<string>

	Where <string> can be any value other than a 0 (zero).

	Args:
		label_data_file (String): Path and name of the file containing label data.

	Returns:
		Pandas DataFrame: Vector containing the label data with 0's and 1's.
	"""
	label_data = []
	print("--Beginning:  Reading label data")
	source_label_data = pandas.read_csv(label_data_file, header=None)
	print("--Finished:  Reading label data")

	# There are two different label files being used:
	# Type 1:
	#	Uses purely numeric values in comma separated file.  A "1"
	#	appears in the 2nd column to indicate malicious/interesting row.
	#
	# Type 2:
	#	Provided by the AIT data set using two columns.  If both column entries
	#	are "0", that row is benign.  If either column contains text, that
	#	row is malicious/interesting.
	#
	# Handle both situations.
	print("--Beginning:  Processing label data")
	for i in source_label_data.index:
		if (type(source_label_data.iloc[i,0]) is np.int64) and (type(source_label_data.iloc[i,0]) is np.int64):
			# If both values are a 0, append 0 indicating non-malicious entry.
			if (source_label_data.iloc[i,0] == 0) and (source_label_data.iloc[i,1] == 0):
				label_data.append(0)
			else:
				label_data.append(1)
		else:
			# We're processing AIT file because the datatype is not an np.int64.
			# label data AND we found a malicious line.
			if (source_label_data.iloc[i,0] == "0") and (source_label_data.iloc[i,1] == "0"):
				label_data.append(0)
			else:
				label_data.append(1)
	print("--Finished:  Processing label data")
	return pandas.DataFrame(label_data)

def process_arguments():
	"""Setup and parse command line arguments using Python's built-in capability.  This includes
	command line help to aid the user in what options are available.

	Returns:
		Dictionary: Command line options and parameters as specified by the user on the command line.
	"""
	# Setup the command line arguments supported by the Python script.
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=available_plugins())
	parser.add_argument("-f", "--file_processor", help="The plugin to use for processing a log file.  Required command line option.", type=str, required=True)
	parser.add_argument("-l", "--log_file", help="The log file (including directory path if needed) to process.  Required command line option.", type=str, required=True)
	parser.add_argument("-m", "--machine_learning", help="The machine learning algorithm to use.  Required command line option.", type=str, required=True)
	parser.add_argument("-n", "--next_machine_learning", help="The second machine learning algorithm to use, using the results from the first.", type=str, required=False)
	parser.add_argument("-s", "--save_path", help="The path and directory to save data generated during execution.", type=str, required=False)
	parser.add_argument("-o", "--output", help="The plugin to use for generating some type of output (graph, etc.).", type=str, required=False)
	parser.add_argument("-a", "--anomaly_reduction", help="The plugin to use for identifying anomalies and reducing log file size.", type=str, required=False)
	parser.add_argument("-d", "--data_labels", help="Labels for the data used in --log_file.", type=str, required=False)
	parser.add_argument("-p", "--plugin_options", help="Supply key=value pair options to a plugin.  See plugin help for supported options.", nargs="*")
	return parser.parse_args()

def save_internal_data(args, raw_data, df):
	"""If the command line option is specified by the user, saves internal data structures as CSV files
	for late examination.  Useful for debugging and seeing what the algorithms are doing with the data 
	through various workflow steps.

	Args:
		args (dictionary): Command line options specified by the user
		raw_data (Pandas DataFrame): Dataframe containing the original data as read from input file.
		df (Pandas DataFrame): Dataframe containing the transformed, machine learning ready version of the data.
	"""
	global ml_results, ml_results2, save_time
	# Grab the command line options used so they can be saved with the rest of the data.
	v = vars(args)
	if (bool(v['plugin_options'])):
		v['plugin_options'] = " ".join(v['plugin_options']) # flatten plugin options array
	ml_results["command_line_options"] = pandas.DataFrame(v,index=[0])

	# If saving, we want to include the raw data as well as the ML ready data.
	ml_results["raw_data"] = raw_data
	ml_results["ml_ready_data"] = df

	# If label data was provided, include that in the output.
	if (args.data_labels):
		ml_results["label_data"] = label_data
	
	# Check to see if a second machine learning algorithm was applied, if so we need to grab
	# that output for saving.
	if (args.next_machine_learning):
		ml_results = {**ml_results, **ml_results2}
	

	# Be sure to capture the time taken to save data to disk as this can contribute to
	# overall execution time.
	start_time = time.time()
	save_data(args.save_path, ml_results)
	save_time = time.time() - start_time

def generate_output(args):
	"""Depending on the output plugin specified by the user, generate output (graph, etc.)
	as part of the workflow.

	Args:
		args (dictionary): Contains command line arguments specified by the user.
	"""
	global ml_results, label_data, output_time
	if "graph" in ml_results:
		start_time = time.time()
		# Typecast the label data to a string.  This automatically triggers plotly to use
		# discrete colors instead of gradient.
		if args.data_labels:
			label_data[0] = label_data[0].astype(str)

		# If KMeans, DBSCAN, or OPTICS was used as the 2nd machine learning algorithm,
		#  grab the centroids to plot them in the plot points colored by label.
		centroids = None
		if "KMeans_Centroids" in ml_results2:
			centroids = ml_results2["KMeans_Centroids"]
		elif "DBSCAN_Centroids" in ml_results2:
			centroids = ml_results2["DBSCAN_Centroids"]
		elif "OPTICS_Centroids" in ml_results2:
			centroids = ml_results2["OPTICS_Centroids"]
		if centroids is not None:
			if (len(label_data) == 0):
				# we have no labels, but we have centroids, build a default label set.
				label_data = pandas.DataFrame(np.full((len(ml_results[ml_results['graph']]),1),"0", dtype = np.str))
			# Loop through and append enough label values for the centroids as provided.
			# pylint: disable=unused-variable
			i = 0
			for centroid in centroids:
				label_data = label_data.append([("Centroid " + str(i))], ignore_index=True)
				i+=1
			# Append the centroids to the main data for plotting.
			ml_results[ml_results['graph']] = np.append(ml_results[ml_results['graph']], centroids, axis=0)

		output[args.output].output(ml_results[ml_results['graph']], label_data)
		output_time = time.time() - start_time
	else:
		print("\n\nWARNING:  Machine learning algorithm does not include graphable results.")
		print("WARNING:  Skipping graph generation.")

def output_time_stats():
	"""Prints basic execution statistics about major steps in the workflow.
	"""
	print("\n\n\n")
	print("-------------------------------------")
	print("\tGENERAL STATISTICS")
	print("File Processing Time: %.4f seconds" % file_processor_time)
	if (args.data_labels):
		print("Data Label Processing Time: %.4f seconds" % label_time)
	print("Machine Learning Time: %.4f seconds" % machine_learning_time)
	if (args.next_machine_learning):
		print("2nd Machine Learning Time: %.4f seconds" % machine_learning_time2)
	if (args.save_path):
		print("Save ML Data Time: %.4f seconds" % save_time)
	if (args.anomaly_reduction):
		print("Anomaly Identification/File Reduction Time: %.4f seconds" % anomaly_reduction_time)
	if (args.output):
		print("Output Generation Time: %.4f seconds" % output_time)
	print("\nTotal Execution Time: %.4f seconds" % (file_processor_time + label_time + machine_learning_time + machine_learning_time2 + save_time + anomaly_reduction_time + output_time))
	print("-------------------------------------\n")

if __name__ == "__main__":
	# Load plugins for reading and processing files
	load_plugins("file_processor", file_processor)
	load_plugins("machine_learning", machine_learning)
	load_plugins("output", output)
	load_plugins("anomaly_reduction", anomaly_reduction)

	# Process command line arguments
	args = process_arguments()

	# Now handle any (optional) plugin command line parameters
	if args.plugin_options:
		for pair in args.plugin_options:
			plugin_option, value = pair.split('=')
			plugin_parameters[plugin_option] = value

	# Based on the user selected plug-in, read the log file into a DataFrame.
	if (args.file_processor) and (args.file_processor in file_processor):
		# Capture start time.
		start_time = time.time()
		# update after changing file processors to take plugin_options argument:
		# raw_data, df = file_processor[args.file_processor].process_file(args.log_file, plugin_parameters)
		raw_data, df = file_processor[args.file_processor].process_file(args.log_file)

		# Capture time taken to process the input file.
		file_processor_time = time.time() - start_time
	else:
		print("\n\nERROR:  Unknown file processor specified: %s\n\n" % args.file_processor)
		exit()

	# See if the user is supplying labels for the data.
	if (args.data_labels):
		# Capture start time.
		start_time = time.time()
		label_data = get_labels(args.data_labels)
		label_time = time.time() - start_time

	if (args.machine_learning) and (args.machine_learning in machine_learning):
		# Capture start time.
		start_time = time.time()
		ml_results = machine_learning[args.machine_learning].machine_learning(df, plugin_parameters)

		# Capture time taken to perform machine learning.
		machine_learning_time = time.time() - start_time
	else:
		print("\n\nERROR:  Unknown machine learning algorithm specified: %s\n\n" % args.machine_learning)
		exit()
	
	# See if the user is chaining multiple machine learning algorithms together
	# (ensemble).  If so, we will use the results from the previous machine learning
	# algorithm as input to the second.
	if (args.next_machine_learning):
		if (args.next_machine_learning in machine_learning):
			# Capture start time.
			start_time = time.time()
			# The first machine learning algorithm in the chain most likely set the "graph" entry to its final
			# output, which identifies the matrix to use as the input for the second algorithm to use as input.
			ml_results2 = machine_learning[args.next_machine_learning].machine_learning(ml_results[ml_results["graph"]], plugin_parameters)

			# Capture time taken to perform machine learning.
			machine_learning_time2 = time.time() - start_time
		else:
			print("\n\nERROR:  Unknown machine learning algorithm specified for second algorithm: %s\n\n" % args.next_machine_learning)
			exit()
	else:
		# User did not specify a 2nd ML algorithm to use, provide empty results.
		ml_results2 = {}

	# See if the user wants to perform anomaly identification/file reduction
	# (not updated for DBSCAN or OPTICS Centroids)
	if (args.anomaly_reduction):
		if (args.anomaly_reduction in anomaly_reduction):
			# Capture start time.
			start_time = time.time()
			if "KMeans_Centroids" in ml_results2:
				if "status" in raw_data:
					results, anomalies = anomaly_reduction[args.anomaly_reduction].anomaly_reduction(ml_results[ml_results['graph']], ml_results2["KMeans_Centroids"], raw_data["status"])
				else:
					results, anomalies = anomaly_reduction[args.anomaly_reduction].anomaly_reduction(ml_results[ml_results['graph']], ml_results2["KMeans_Centroids"])

			# Capture time taken to perform machine learning.
			anomaly_reduction_time = time.time() - start_time

			ml_results["anomaly_reduction_results"] = results
			ml_results["identified_anomalies"] = anomalies
		else:
			print("\n\nERROR:  Unknown anomaly/reduction plugin specified: %s\n\n" % args.anomaly_reduction)
			exit()

	# Check to see if the user wants to save data generated during the machine learning
	# part of the workflow.
	if (args.save_path):
		save_internal_data(args, raw_data, df)

	# Check to see if the user wants some type of output generated during execution.
	if (args.output):
		# If so (because it's optional), make sure the output plugin exists.
		if (args.output in output):
			generate_output(args)
		else:
			print("\n\nERROR:  Unknown output plugin specified: %s\n\n" % args.output)
			exit()

	# Output basic execution stats.
	output_time_stats()
