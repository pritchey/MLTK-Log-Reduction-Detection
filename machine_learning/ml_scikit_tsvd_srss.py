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
	with_mean = True

	# The following are used to tune where the split point occurs in the columns (features)
	# for reduction down to 2 columns (features) for final output.  Also, the maximum
	# number of columns (features) to use.
	srss_split = 1
	srss_max = 3

	fitting_time = 0
	srss_time = 0

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		text = "\tName: ml_scikit_tsvd_srss"
		text += "\n\t\tThis machine learning plugin uses scikit-learn's truncated SVD algorithm and applies square root sum of squares to the results to reduce the results to 2 dimensions.\n"
		text += "\n\t\tOptional Parameters:"
		text += "\n\t\t\tsrss_split:  Column number to split the input dataframe at for calculating square root of square sums."
		text += "\n\t\t\tsrss_max:  Maximum number of columns in the input dataframe to use."
		text += "\n\t\t\ttsvdsrss_skip_normalization:  Do NOT perform normalization (scaling) of data, skip this step."
		return text

	def calculate_srss(self, df):
		"""Reduces the passed in matrix to a 2d matrix using square root of sum of squares on groups of columns.

		Returns:
			Numpy Array: Matrix to perform the reduction on.
		"""
		# Split the matrix based upon tuning paramters and maximum number of columns to use
		# and perform the calculation on each column group to reduce it to a single column.
		columna = np.sqrt(np.sum(np.square(df[:, :self.srss_split]), axis=1))
		columnb = np.sqrt(np.sum(np.square(df[:, self.srss_split:self.srss_max]), axis=1))

		# Take the generated columns, transpose and concatenate together to form the final matrix result.
		final_df = np.concatenate((columna[np.newaxis].T, columnb[np.newaxis].T), axis = 1)
		return final_df

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
		print("\tMachine learning algorithm:  scikit-learn TSVD with square root of sum of squares")
		# If the user specified tuning options on the command line, override the built-ins.
		#
		if "srss_split" in plugin_options:
			self.srss_split = int(plugin_options["srss_split"])
			print("\tOverriding default srss_split, it is set to: %i" % self.srss_split)
		else:
			print("\tUsing default setting for srss_split: %i" % self.srss_split)

		if "srss_max" in plugin_options:
			self.srss_max = int(plugin_options["srss_max"])
			print("\tOverriding default srss_max, it is set to: %i" % self.srss_max)
		else:
			print("\tUsing default setting for srss_max: %i" % self.srss_max)

		# In order to save the full US matrix as calculated by TruncatedSVD, we need to add
		# a column of zeros due to an API limitation that prevents us from retaining all columns.
		if self.srss_max == df.shape[1]:
			print("\tUsing fix to retain all columns due to number of srss_max setting...")
			df["Zero"] = 0

		# Capture start time.
		start_time = time.time()

		# Create an instance of the Truncated SVD, a normalizer, and create a pipeline for
		# automatic execution of both.
		svd = TruncatedSVD(self.srss_max,random_state=self.random_state,n_iter=self.n_iter)
		normalizer = StandardScaler(copy=False,with_mean=self.with_mean,with_std=True)

		# Build out the pipeline depending if the user opted to bypass normalization
		if "tsvdsrss_skip_normalization" not in plugin_options:
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

		print("\tBeginning:  calculating square root of sum of squares")
		start_time_srss = time.time()
		srss_reduction = self.calculate_srss(US)
		self.srss_time = time.time() - start_time_srss
		print("\tFinished:  calculating square root of sum of squares")

		print("\n\tFitting Time: %.4f seconds" % self.fitting_time)
		print("\tSRSS Calculation Time: %.4f seconds" % self.srss_time)
		print("\tMachine Learning Total Time: %.4f seconds" % (time.time() - start_time))

		print("--Finished:  Machine Learning")

		# Return a dictionary containing specific components created or calculated
		# as part of the machine learning process.  These may be used to perform
		# additional tasks (saving data to files, graphing, etc.).
		return {"US": US, "S": S, "VT": VT, "T": T, "graph": "srss_reduction", "Variance_Ratios": variance, "srss_reduction": srss_reduction}
