import time # Imported so execution time can be captured
from pympler.asizeof import asizeof # Used to get more accurate memory utilization
from pandas import DataFrame # Using Pandas dataframes for performance and code clarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class simPlugin(object):
	state = ""
	random_state = 12345
	n_iter = 5
	n_components = 2
	with_mean = True

	fitting_time = 0

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: ml_scikit_tsvd"
		text += "\n\t\tThis machine learning plugin uses scikit-learn's truncated SVD algorithm.\n"
		text += "\n\t\tOptional Parameters:"
		text += "\n\t\t\ttsvd_skip_normalization:  Do NOT perform normalization (scaling) of data, skip this step."
		text += "\n\t\t\ttsvd_number_of_components:  Specify the number of components to retain (default is 2)."

		return text

	def machine_learning(self, df, plugin_options):
		"""Apply the scikit-learn truncated singular value decomposition (TSVD) machine learning algorithm to the supplied data set.

		Args:
			df (Pandas DataFrame): DataFrame containing the machine learning ready version of the dataset to be processed.
			plugin_options (dictionary):  Dictionary containing any optional parameters for plugins being used.

		Returns:
			Dictionary: Dictionary containing final machine learning results and other internal data that user may want to save for review.
		"""
		print("\n")
		print("--Beginning:  Machine Learning")
		print("\tMachine learning algorithm:  scikit-learn TSVD")

		if ("tsvd_number_of_components" in plugin_options):
			self.n_components = int(plugin_options["tsvd_number_of_components"])
			print("\tOverriding default number of components, it is set to: %i" % self.n_components)
		else:
			print("\tUsing default setting for number of components: %i" % self.n_components)

		#---------BEGIN TEMPORARY CODE---------
		# In order to save the full US matrix as calculated by TruncatedSVD, we need to add
		# a column of zeros due to an API limitation that prevents us from retaining all columns.
		# This is temporary to allow review of the calculated outputs. This code should be remove
		# prior to executing final experiments.
		if self.n_components == df.shape[1]:
			print("\tUsing fix to retain all columns due to number of components setting...")
			df["Zero"] = 0
			#self.n_components = df.shape[1] -1
		#-----------END TEMPORARY CODE----------

		# Capture start time.
		start_time = time.time()

		# Create an instance of the Truncated SVD, a normalizer, and create a pipeline for
		# automatic execution of both.
		svd = TruncatedSVD(self.n_components,random_state=self.random_state,n_iter=self.n_iter)
		normalizer = StandardScaler(copy=False,with_mean=self.with_mean,with_std=True)

		# Build out the pipeline depending if the user opted to bypass normalization
		if "tsvd_skip_normalization" not in plugin_options:
			lsa = make_pipeline(normalizer, svd)
		else:
			print("\t\tNOT normalizing data as requested by user...")
			lsa = make_pipeline(svd)

		print("\tBeginning:  fitting")
		start_time_fitting = time.time()
		# In order to perform a diff for sanity checking, we must typecast the 
		# Pandas dataframe to a Numpy array even though "copy=false" is set for the
		# normalizer.  The results will otherwise be incorrect.
		df_np = df.to_numpy()
		US = lsa.fit_transform(df_np)
		self.fitting_time = time.time() - start_time_fitting
		print("\tFinished:  fitting")

		S = svd.singular_values_
		VT = svd.components_
		variance = svd.explained_variance_ratio_
		
		print("\nU*S =\n", US)
		print("\nS =\n", S)
		print("\nVT =\n", VT)
		
		AA = US @ VT # np.matmul(US,VT) # reconstruct A
		print("\nAA =\n", AA)
		print("\ndiff =", np.linalg.norm(AA-df_np))

		T = S/S[0]
		T *= T
		T = np.cumsum(T) / sum(T)
		print("\nT =\n", T)

		print("\n\tFitting Time: %.4f seconds" % self.fitting_time)
		print("\tMachine Learning Total Time: %.4f seconds" % (time.time() - start_time))

		print("--Finished:  Machine Learning")

		# Return a dictionary containing specific components created or calculated
		# as part of the machine learning process.  These may be used to perform
		# additional tasks (saving data to files, graphing, etc.).
		return {"US": US, "S": S, "VT": VT, "T": T, "graph": "US", "Variance_Ratios": variance}
