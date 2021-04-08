from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class simPlugin(object):

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: ar_euclidean_distance\n\t\tPerform anomaly identification and log reduction using euclidean distance.\n"

	def classify(self, x):
		"""Determine the label (0 or 1) depending on which centroid the plot point is closest to.

		Args:
			x (Array): Array containing the calculated euclidean distances to the two centroids.

		Returns:
			Integer: A 0 or a 1 depending on which centroid the plot point is closest to.
		"""
		if (x[0] < x[1]):
			return 0
		else:
			return 1

	def identify_classes(self, results):
		"""Using the results from calculating the euclidean distances, iterate over the results and assign a label.

		Args:
			results (Pandas DataFrame): Contains the ML algorithms final results.

		Returns:
			Pandas DataFrame: DataFrame containing labels for the dataset.
		"""
		classes = np.array([])
		classes = np.append(classes, np.apply_along_axis(self.classify, axis=1, arr=results))
		return classes

	def anomaly_reduction(self, df1, df2):
		"""Using the machine learning results, identify anomalies which can then be leveraged to perform log file
		data reduction.  This plugin uses the euclidean distance to determine which centroid a vector is closest to.

		NOTE:  This version assumes there are only two (2) centroids.

		Args:
			df1 (Pandas DataFrame): DataFrame containing either the ML results to use in the calculation.
			df2 (Pandas DataFrame): DataFrame containing either the centroids to use in the calculation.

		Returns:
			[type]: [description]
		"""
		print("\n")
		print("--Beginning:  Euclidean distance anomaly identification/file reduction")

		print("\tBeginning:  Calculating euclidean distances...")
		results = []
		for i in df1:
			results.append([np.linalg.norm(i - df2[0]), np.linalg.norm(i - df2[1])])
		print("\tFinished:  Calculating euclidean distances...")

		print("\tBeginning:  Beginning anomaly identification...")
		anomaly_results = self.identify_classes(results)
		print("\tFinished:  Finished anomaly identification...")

		print("--Finished:  Euclidean distance anomaly identification/file reduction\n\n")
		return results, anomaly_results