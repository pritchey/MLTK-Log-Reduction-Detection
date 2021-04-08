import time # Imported so execution time can be captured
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame # Using Pandas dataframes for performance and code clarity
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class simPlugin(object):
	state = ""
	with_mean = True
	eps = 1.0
	min_samples = 5
	fitting_time = 0

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: ml_scikit_OPTICS"
		text += "\n\t\tThis machine learning plugin uses scikit-learn's OPTICS algorithm.\n"
		text += "\n\t\tOptional Parameters:"
		text += "\n\t\t\tOPTICS_skip_normalization:  Do NOT perform normalization (scaling) of data, skip this step."
		text += "\n\t\t\OPTICS_eps:  Specify eps parameter (default is 1.0)."
		text += "\n\t\t\OPTICS_min_samples:  Specify min_samples parameter (default is 5)."
#
# OPTICS (with memory complexity n) is an alternative to DBSCAN (with memory complexity n^2)
# which has time complexity n^2 in general with the default max_eps = np.inf. 
# We will set max_eps = eps to reduce the run-time.
#
		return text

	def machine_learning(self, df, plugin_options):
		"""Apply the scikit-learn OPTICS machine learning algorithm to the supplied data set, returning the results and indices.

		Args:
			df (Pandas DataFrame): DataFrame containing the machine learning ready version of the dataset to be processed.
			plugin_options (dictionary):  Dictionary containing any optional parameters for plugins being used.

		Returns:
			Dictionary: Dictionary containing final machine learning results and other internal data that user may want to save for review.
		"""
		print("\n")
		print("--Beginning:  Machine Learning")
		print("\tMachine learning algorithm:  scikit-learn OPTICS")

		if ("OPTICS_eps" in plugin_options):
			self.eps = float(plugin_options["OPTICS_eps"])
			print("\tOverriding default eps, it is set to: %g" % self.eps)
		else:
			print("\tUsing default setting for eps: %g" % self.eps)

		if ("OPTICS_min_samples" in plugin_options):
			self.min_samples = int(plugin_options["OPTICS_min_samples"])
			print("\tOverriding default min_samples, it is set to: %i" % self.min_samples)
		else:
			print("\tUsing default setting for min_samples: %g" % self.min_samples)

		# Capture start time.
		start_time = time.time()

		# Create an instance of OPTICS, a normalizer, and create a pipeline for
		# automatic execution of both.
		# cluster_method = "xi" or "dbscan"
		optics = OPTICS(max_eps = self.eps, min_samples = self.min_samples, cluster_method = "xi")
		normalizer = StandardScaler(copy=False,with_mean=self.with_mean,with_std=True)

		print("\tBeginning:  fitting")
		start_time_fitting = time.time()

		# Check to see if the user wants to skip normalization of the data before
		# applying OPTICS to the data.
		if "OPTICS_skip_normalization" not in plugin_options:
			normalized_data = normalizer.fit_transform(df)
		else:
			print("\t\tNOT normalizing data as requested by user...")
			normalized_data = df
		optics.fit(normalized_data)
		results = cluster_optics_dbscan(reachability = optics.reachability_, core_distances = optics.core_distances_, ordering = optics.ordering_, eps = self.eps)

		# Number of clusters in labels, ignoring noise if present.
		n_clusters = len(set(results)) - (1 if -1 in results else 0)
		n_noise = list(results).count(-1)
		print("\tn_clusters = %i, n_noise = %i" % (n_clusters, n_noise) )

		# compute centroids of the clusters
		centroids = np.zeros((n_clusters,normalized_data.shape[1]))
		for i in range(0,n_clusters):
			j = [k for k, x in enumerate(results) if x == i]
			centroids[i] = np.sum(normalized_data[j],axis=0)/len(j)

		self.fitting_time = time.time() - start_time_fitting
		print("\tFinished:  fitting")

		print("\n\tFitting Time: %.4f seconds" % self.fitting_time)
		print("\tMachine Learning Total Time: %.4f seconds" % (time.time() - start_time))

		print("--Finished:  Machine Learning")

		# Return a dictionary containing specific components created or calculated
		# as part of the machine learning process.  These may be used to perform
		# additional tasks (saving data to files, graphing, etc.).
		return {"OPTICS_Results": results, "OPTICS_Centroids": centroids}
