import time # Imported so execution time can be captured.
import pandas as pd # Using Pandas dataframes for performance and code clarity.
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
color = sns.color_palette()

class simPlugin(object):
	state = ""

	def __init__(self):
		self.state = "Initialized"
    
	def print_help(self):
		"""Builds a formatted help text string for the plugin that's displayed when the user specifies the help command line option.

		Returns:
			String: Formatted text string containing the plugin's help information.
		"""
		return "\tName: out_seaborn_3d_scatter\n\t\tGenerates an interactable 3d scatter plot of data using the seaborn library.\n"

	def output(self, ml_results, label_data):
		"""Generate a 3D scatter plot using the Seaborn graphing library.

		Args:
			ml_results (Pandas DataFrame): DataFrame containing the output from a machine learning algorithm to be plotted.
			label_data (Pandas DataFrame): Optional DataFrame containing label data (0's and 1's)
		"""
		log_line = []
		print("\n")
		print("--Beginning:  Seaborn 3D scatter plot generation")

		# Ensure we have a Pandas data frame and carve off the first two columns
		# we're only doing a 2d graph.
		df = pd.DataFrame(ml_results)
		df = df.iloc[:, :3]

		# We need to be able to trace a single plot point back to a specific log line,
		# build an array of strings with the necessary information.
		for i in range(len(df.index)):
			log_line.append("Log Line: " + str(i + 1))

		# We may or may not have been passed data labels.  Typecast to ensure it's a Pandas
		# data frame, then check the size.  If we have labels, concatenate it.
		labels = pd.DataFrame(label_data)
		tempDF = pd.DataFrame(data=df.loc[:,0:2], index=df.index)

		fig = plt.figure()
		ax = Axes3D(fig)

		if labels.size > 0:
			tempDF = pd.concat((tempDF,labels), axis=1, join="inner")
			tempDF.columns = ["X", "Y", "Z", "Label"]
			graph = ax.scatter(tempDF['X'], tempDF['Y'], tempDF['Z'], c=tempDF["Label"], s=60)
			legend = ax.legend(*graph.legend_elements(), loc="lower center", title="Labels", borderaxespad=-10, ncol=4)
			ax.add_artist(legend)
		else:
			tempDF.columns = ["X", "Y", "Z"]
			ax.scatter(tempDF['X'], tempDF['Y'], tempDF['Z'], c='skyblue', s=60)
		
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_zlabel("Z")
		plt.show()

		print("--Finished:  Seaborn 3D scatter plot generation")