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
		return "\tName: ar_cosine_similarity_apache\n\t\tPerform anomaly identification and log reduction using cosine similarity.\n"

	def classify(self, x):
		"""Determine the label (0 or 1) depending on the calculated cosine similarities between the plot point and the centroids.

		Args:
			x (Array): Array containing the calculated cosine similarities.

		Returns:
			Integer: A 0 or a 1 depending on which centroid the plot point most similar to.
		"""
		if (x[0] > x[1]):
			return 0
		else:
			return 1

	def identify_classes(self, results, statusCode):
		"""Using the results from calculating the euclidean distances, iterate over the results and assign a label.
		If needed, flip the labelling.

		Args:
			results (Pandas DataFrame): 
			statusCode (Pandas DataFrame):  Contains the logged Apache request status code.

		Returns:
			Pandas DataFrame: DataFrame containing labels for the dataset.
		"""
		classes = np.empty_like(results)

		for i, x in enumerate(results):
			classes[i] = self.classify(x)

		return classes

	def anomaly_reduction(self, df1, df2, statusCode):
		"""Using the machine learning results, identify anomalies which can then be leveraged to perform log file
		data reduction.  This plugin uses the cosine similarity to determine which centroid a vector is most similar to.
		Uses the Apache status code to determine if the labelling needs to be flipped due to the plot point
		cluster/KMeans centroid orientation.

		NOTE:  This version assumes there are only two (2) centroids.

		Args:
			df1 (Pandas DataFrame): DataFrame containing either the ML results to use in the calculation.
			df2 (Pandas DataFrame): DataFrame containing either the centroids to use in the calculation.
			statusCode (Pandas DataFrame): DataFrame containing the Apache status code for each row in the dataset.

		Returns:
			[type]: [description]
		"""
		print("\n")
		print("--Beginning:  Cosine similarity anomaly identification/file reduction")

		print("\tBeginning:  Calculating cosine similarity...")
		results = cosine_similarity(df1, df2)
		print("\tFinished:  Calculating cosine similarity...")

		print("\tBeginning:  Beginning anomaly identification...")
		anomaly_results = self.identify_classes(results, statusCode)
		print("\tFinished:  Finished anomaly identification...")

		print("--Finished:  Cosine similarity anomaly identification/file reduction\n\n")
		return results, anomaly_results