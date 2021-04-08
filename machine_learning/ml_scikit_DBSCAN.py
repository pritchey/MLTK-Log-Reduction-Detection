import time # Imported so execution time can be captured
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame # Using Pandas dataframes for performance and code clarity
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class simPlugin(object):
	state = ""
	with_mean = True
	eps = 1.0
	min_samples = 5
	rtol = 0.0 # relative tolerance for combining duplicate samples
	fitting_time = 0

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: ml_scikit_DBSCAN"
		text += "\n\t\tThis machine learning plugin uses scikit-learn's DBSCAN algorithm.\n"
		text += "\n\t\tOptional Parameters:"
		text += "\n\t\t\tDBSCAN_rtol:  Specify rtol parameter (default is 0.0)."
		text += "\n\t\t\tDBSCAN_eps:  Specify eps parameter (default is 1.0)."
		text += "\n\t\t\tDBSCAN_min_samples:  Specify min_samples parameter (default is 5)."
		text += "\n\t\t\tDBSCAN_skip_normalization:  Do NOT perform normalization (scaling) of data, skip this step."

		return text

	def machine_learning(self, df, plugin_options):
		"""Apply the scikit-learn DBSCAN machine learning algorithm to the supplied data set, returning the results and indices.

		Args:
			df (Pandas DataFrame): DataFrame containing the machine learning ready version of the dataset to be processed.
			plugin_options (dictionary):  Dictionary containing any optional parameters for plugins being used.

		Returns:
			Dictionary: Dictionary containing final machine learning results and other internal data that user may want to save for review.
		"""
		print("\n")
		print("--Beginning:  Machine Learning")
		print("\tMachine learning algorithm:  scikit-learn DBSCAN")

		# If the user specified tuning options on the command line, override the built-ins.
		#
		if "DBSCAN_rtol" in plugin_options:
			self.rtol = float(plugin_options["DBSCAN_rtol"])
			print("\tOverriding default rtol, it is set to: %g" % self.rtol)
		else:
			print("\tUsing default setting for rtol: %g" % self.rtol)

		if "DBSCAN_eps" in plugin_options:
			self.eps = float(plugin_options["DBSCAN_eps"])
			print("\tOverriding default eps, it is set to: %g" % self.eps)
		else:
			print("\tUsing default setting for eps: %g" % self.eps)

		if "DBSCAN_min_samples" in plugin_options:
			self.min_samples = int(plugin_options["DBSCAN_min_samples"])
			print("\tOverriding default min_samples, it is set to: %i" % self.min_samples)
		else:
			print("\tUsing default setting for min_samples: %i" % self.min_samples)

		# Capture start time.
		start_time = time.time()

		# Create an instance of DBSCAN, a normalizer, and create a pipeline for
		# automatic execution of both.
		dbscan = DBSCAN(eps = self.eps, min_samples = self.min_samples)
		normalizer = StandardScaler(copy=False,with_mean=self.with_mean,with_std=True)

		print("\tBeginning:  fitting")
		start_time_fitting = time.time()

		# Check to see if the user wants to skip normalization of the data before
		# applying DBSCAN to the data.
		if "DBSCAN_skip_normalization" not in plugin_options:
			normalized_data = normalizer.fit_transform(df)
		else:
			print("\t\tNOT normalizing data as requested by user...")
			normalized_data = df

		sample_counts = None

		# if relative tolerance was specified, combine duplicate samples
		if self.rtol:
			print("\tSorting and combining...")
			s = np.lexsort(np.fliplr(normalized_data).T)
			r = []; sample_counts = []; j = -1 # r[j] will be current entry
			for i in s:
				cur = normalized_data[i]
				if len(r) == 0 or np.linalg.norm(cur-prev) > tol:
					r.append(cur) # distinct sample
					sample_counts.append(1)
					prev = cur
					tol = self.rtol*(np.linalg.norm(cur) + self.rtol)
					j += 1
				else: # duplicate
					sample_counts[j] += 1;
			normalized_data = np.array(r)
			print("\tReduced data dimensions: %i x %i" % (len(normalized_data[0]), len(normalized_data)))

		dbscan.fit(normalized_data,sample_weight=sample_counts)
		results = dbscan.labels_
		core_indices = dbscan.core_sample_indices_

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

		# if rtol reduction was performed, return extra values: normalized_data, sample_counts, and "graph"
		if self.rtol:
			return {"DBSCAN_Results": results, "DBSCAN_Core_Indices": core_indices, "DBSCAN_Centroids": centroids, "DBSCAN_Normalized_Data": normalized_data, "DBSCAN_Sample_Counts": sample_counts, "graph": "DBSCAN_Normalized_Data"}
		else:
			return {"DBSCAN_Results": results, "DBSCAN_Core_Indices": core_indices, "DBSCAN_Centroids": centroids}
