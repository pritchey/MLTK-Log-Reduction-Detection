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
		return "\tName: ar_euclidean_distance_apache\n\t\tPerform anomaly identification and log reduction using euclidean distance.  Attempts to properly orient data based on appropriate centroid.\n"

	def flip_classes(self, x):
		"""Flip the passed in label from 0 to 1 or 1 to 0.

		Args:
			x (Array): Array containing a single value indicating which label was assigned (0 or 1).

		Returns:
			Integer: A 0 or a 1 after flipping the value.
		"""
		return (x[0] + 1) % 2

	def classify(self, x, statusCode):
		"""Determine the label (0 or 1) depending on which centroid the plot point is closest to.

		Args:
			x (Array): Array containing the calculated euclidean distances to the two centroids.
			statusCode(Numeric): Floating point number indicating type of Apache status code seen on the log line

		Returns:
			Integer: A 0 or a 1 depending on which centroid the plot point is closest to.
			Integer: 0 if status code was OK, otherwise 2 or 1 depending on which centroid the error code was closer too.
		"""
		if (x[0] < x[1]):
			if statusCode == 0.2:
				return 0, 0
			else:
				return 0, 2
		else:
			if statusCode == 0.2:
				return 1, 0
			else:
				return 1, 1

	def identify_classes(self, results, statusCode):
		"""Using the results from calculating the euclidean distances, iterate over the results and assign a label.
		Additionally, automatically detect if the centroids/label assignments are reversed based on where Apache error
		codes are predominantly assigned.  If needed, flip the labeling.

		Args:
			results (Pandas DataFrame): Contains the ML algorithms final results.
			statusCode (Pandas DataFrame):  Contains the logged Apache request status code.

		Returns:
			Pandas DataFrame: DataFrame containing labels for the dataset.
		"""
		#classes = np.array([])
		classes = np.empty_like(results)

		# Unfortuantely, needing to utilize 2 arrays simultaneously prevents us from using
		# the Numpy nataive API.
		#classes = np.append(classes, np.apply_along_axis(self.classify, axis=1, arr=results))
		centroidzero = 0
		centroidone = 0
		for i, x in enumerate(results):
			classes[i], errorstatuscentroid = self.classify(x, statusCode[i])
			# Track how many non-good status codes are on centroid zero.  If there's too many,
			# we need to flip labelling due to where the centroids/clusters ended up.
			# (Things were labled backwards)
			if errorstatuscentroid == 1:
				centroidzero += 1
			elif errorstatuscentroid == 2:
				centroidone += 1

		print("\t\tCounter Centroid Zero: %i" % centroidzero)
		print("\t\tCounter Centroid One: %i" % centroidone)

		if centroidzero > centroidone:
			print("\t\tCENTROIDS REVERSED:  flipping labels")
			classes = np.apply_along_axis(self.flip_classes, axis=1, arr=classes)
		return classes

	def anomaly_reduction(self, df1, df2, statusCode):
		"""Using the machine learning results, identify anomalies which can then be leveraged to perform log file
		data reduction.  This plugin uses the euclidean distance to determine which centroid a vector is closest to.
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
		print("--Beginning:  Euclidean distance anomaly identification/file reduction")

		print("\tBeginning:  Calculating euclidean distances...")
		results = []
		for i in df1:
			results.append([np.linalg.norm(i - df2[0]), np.linalg.norm(i - df2[1])])
		print("\tFinished:  Calculating euclidean distances...")

		print("\tBeginning:  Beginning anomaly identification...")
		anomaly_results = self.identify_classes(results, statusCode)
		print("\tFinished:  Finished anomaly identification...")

		print("--Finished:  Euclidean distance anomaly identification/file reduction\n\n")
		return results, anomaly_results