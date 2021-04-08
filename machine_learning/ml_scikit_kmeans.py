import time # Imported so execution time can be captured
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame # Using Pandas dataframes for performance and code clarity
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class simPlugin(object):
	state = ""
	random_state = 12345
	n_iter = 5
	n_clusters = 2
	with_mean = True

	fitting_time = 0

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: ml_scikit_kmeans"
		text += "\n\t\tThis machine learning plugin uses scikit-learn's kmeans algorithm.\n"
		text += "\n\t\tOptional Parameters:"
		text += "\n\t\t\tkmeans_skip_normalization:  Do NOT perform normalization (scaling) of data, skip this step."
		text += "\n\t\t\kmeans_number_of_clusters:  Specify the number of clusters to retain (default is 2)."

		return text

	def machine_learning(self, df, plugin_options):
		"""Apply the scikit-learn KMeans machine learning algorithm to the supplied data set, returning the results and centroids.

		Args:
			df (Pandas DataFrame): DataFrame containing the machine learning ready version of the dataset to be processed.
			plugin_options (dictionary):  Dictionary containing any optional parameters for plugins being used.

		Returns:
			Dictionary: Dictionary containing final machine learning results and other internal data that user may want to save for review.
		"""
		print("\n")
		print("--Beginning:  Machine Learning")
		print("\tMachine learning algorithm:  scikit-learn K-means")

		if ("kmeans_number_of_clusters" in plugin_options):
			self.n_components = int(plugin_options["kmeans_number_of_clusters"])
			print("\tOverriding default number of clusters, it is set to: %i" % self.n_clusters)
		else:
			print("\tUsing default setting for number of clusters: %i" % self.n_clusters)

		# Capture start time.
		start_time = time.time()

		# Create an instance of the Truncated SVD, a normalizer, and create a pipeline for
		# automatic execution of both.
		kmeans = KMeans(n_clusters = self.n_clusters, random_state=self.random_state)
		normalizer = StandardScaler(copy=False,with_mean=self.with_mean,with_std=True)

		print("\tBeginning:  fitting")
		start_time_fitting = time.time()

		# Check to see if the user wants to skip normalization of the data before
		# applying KMeans to the data.
		if "kmeans_skip_normalization" not in plugin_options:
			normalized_data = normalizer.fit_transform(df)
		else:
			print("\t\tNOT normalizing data as requested by user...")
			normalized_data = df
		kmeans.fit(normalized_data)
		kmeans.predict(normalized_data)
		results = kmeans.labels_
		centroids = kmeans.cluster_centers_

		self.fitting_time = time.time() - start_time_fitting
		print("\tFinished:  fitting")

		print("\n\tFitting Time: %.4f seconds" % self.fitting_time)
		print("\tMachine Learning Total Time: %.4f seconds" % (time.time() - start_time))

		print("--Finished:  Machine Learning")

		# Return a dictionary containing specific components created or calculated
		# as part of the machine learning process.  These may be used to perform
		# additional tasks (saving data to files, graphing, etc.).
		return {"KMeans_Results": results, "KMeans_Centroids": centroids}