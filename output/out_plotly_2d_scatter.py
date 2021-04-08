import time # Imported so execution time can be captured.
import pandas as pd # Using Pandas dataframes for performance and code clarity.
import plotly.express as px

class simPlugin(object):
	state = ""

	def __init__(self):
		self.state = "Initialized"

	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: out_plotly_2d_scatter\n\t\tGenerates an interactable 2d scatter plot of data using the plotly library.\n"

	def output(self, ml_results, label_data):
		"""Generate a 2D scatter plot using the Plotly graphing library.

		Args:
			ml_results (Pandas DataFrame): DataFrame containing the output from a machine learning algorithm to be plotted.
			label_data (Pandas DataFrame): Optional DataFrame containing label data (0's and 1's)
		"""
		log_line = []
		print("\n")
		print("--Beginning:  Plotly 2d scatter plot generation")

		# Ensure we have a Pandas data frame and carve off the first two columns
		# we're only doing a 2d graph.
		df = pd.DataFrame(ml_results)
		df = df.iloc[:, :2]

		# We need to be able to trace a single plot point back to a specific log line,
		# build an array of strings with the necessary information.
		for i in range(len(df.index)):
			log_line.append("Log Line: " + str(i + 1))

		# We may or may not have been passed data labels.  Typecast to ensure it's a Pandas
		# data frame, then check the size.  If we have labels, concatenate it.
		labels = pd.DataFrame(label_data)
		if labels.size > 0:
			df = pd.concat([df, labels], axis=1, ignore_index=True)
			df.columns = ["x", "y", "Label"]
			# In the event a user is chaining multiple algorithms, one of which calculates centroids and
			# does not have labels to go with the ML results data we need to fill all NaN (not a null)
			# with zeros.  Otherwise the plotting will fail.
			df["Label"] = df["Label"].fillna(0)
			fig = px.scatter(df, x="x", y="y", color="Label", hover_name=log_line, color_discrete_sequence=px.colors.qualitative.Safe)
		else:
			df.columns = ["x", "y"]
			fig = px.scatter(df, x="x", y="y", hover_name=log_line, color_discrete_sequence=px.colors.qualitative.Safe)
		fig.show()

		print("--Finished:  Plotly 2d scatter plot generation")