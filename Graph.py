import csv
from genericpath import isfile
from logging.config import valid_ident
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import json
from typing import Callable, TypeVar, Generic
from matplotlib.ticker import FormatStrFormatter
import copy
import re
import numpy as np
import matplotlib as mpl
from time import sleep

def reduce(eval, base, combine, arr):
	"""This is used to reduce an array to single value

	Args:
		eval (a -> b function): A function that takes in a value of type a, and returns a value of type b
		base (b): A "base" value of type b, when the list is reduced to []
		combine (b, b -> b function) A function used to combine two values
		arr (a list): A list of values of type a to reduce

	Returns:
		b: The value that the array has been reduced to
	"""
	match(arr):
		case []:
			return base
		case [a]:
			return eval(a)
		case [a, *rest]:
			return combine(eval(a), reduce(eval, base, combine, rest))
		
def JoinLast(join: str, arr: list[any], last: str = None) -> str:
	"""Similar to the string.join function, but allows for the last two items to be joined together with a different string

	Args:
		join (str): The string to join array items
		arr (list[any]): Array of items to be joined
		last (str, optional): A string to join the last two array items. Defaults to None. When value is None, the last two items are joined using `join`.

	Returns:
		str: The resultant string after being joined by `join` and `last`.
	"""
	if last == None: last = join
	return arr[0] if len(arr) == 0 else join.join(arr[:len(arr)-1]) + last + arr[len(arr) - 1]

def IsDigit(s: str) -> bool:
	"""Determines if a string passed in represents a digit. The built in function does not support negative numbers.

	Args:
		s (string): a string, possibly representing a number

	Returns:
		bool: Is the string a number?
	"""
	return s.lstrip("-").isdigit()

class Value:
	"""A way of abstracting each value the user might want to input to their preset.
	Avoids code duplication, and makes it easier to add/remove values in the future.
	"""
	def __init__(self, name: str, newPrompt: str, modifyPrompt: str, modifyCommand: str, help: str, display: str, verify: Callable[[str], bool], convert: Callable[[str], any], value: any, isMandatory: bool, advancedOption: bool = False, restrictType = []) -> None:
		"""Initializes the Value

		Args:
			name (str): The name of the value
			newPrompt (str): The prompt the user sees when they are initializing a new value
			modifyPrompt (str): The prompt the user sees when all the modification functions are displayed
			modifyCommand (str): The command used to change the value
			help (str): Displays more information about the purpose or type of the value
			display (str): Displays the name and the value associated with the Value
			verify (Callable[[str], bool]): A function that takes the string input and returns a bool, determines if the input was valid or not
			value (str): The actual value being stored
			isMandatory (bool): is this value mandatory, or can the default be used?
			advancedOption (bool): Certain options are hidden by default to reduce clutter, should this value be hidden?
			restrictType (list[str]): Restrict the value to only certain types. List of types, empty list defaults as no restriction.
		"""
		self.name = name
		self.newPrompt = newPrompt
		self.modifyPrompt = modifyPrompt
		self.modifyCommand = modifyCommand
		self.help = help
		self.display = display
		self.verify = verify
		self.convert = convert
		self.value = value
		self.isMandatory = isMandatory
		self.advancedOption = advancedOption
		self.restrictType = restrictType

"""Different types of plots the function supports"""
AxisTypes = [
	"line",
	"scatter",
	"Ramachandran",
	"gradient"
]

"""Different settings related to each individual figure the user can change"""
AxisValues = {
	"name": Value("name", "What should the name of your preset be? ", "name [New Name]", "name ",
		"The name of your preset. Only used to help identify and distinguish from other presets.", "Name: ", lambda name: type(name) == str, lambda name: name, "", True),
	"comment": Value("comment", "Optionally leave a helpful comment to yourself about the purpose of this preset.", "comment [Comment]", "comment ",
		"Just something to help you remember what this is used for when you come back to it later. Has no effect on the behavior of the preset.", "Comment: ", lambda comment: type(comment) == str, lambda comment: comment, "", False),
	"type": Value("type", f"What type plot are you trying to make? {JoinLast(', ', AxisTypes, ', or ')} ", f"type [{'|'.join(AxisTypes)}]", "type ", 
	    f"The type of plot you want to make with your data. The options are {JoinLast(', ', AxisTypes, ', and')}.", "Type: ", lambda t: t in AxisTypes, lambda t: t, "", True, False),
	"numRows": Value("numRows", "How many rows are in the largest file? ", "numRows [Number of Rows]", "numRows ",
		"The number of rows in the largest file you want to graph. This just helps speed up read-in of the files.", "Number of Rows: ", lambda numRows: numRows.isdigit(), lambda numRows: int(numRows), 0, True),
	"numPlots": Value("numPlots", "How many datasets are you looking to plot? ", "numPlots [Number of Datasets]", "numPlots ",
		"The number of sets of data you want to graph; the number of files you want to graph.", "Number of Plots: ", lambda numPlots: numPlots.isdigit(), lambda numPlots: int(numPlots), 0, True),
	"xAxisTitle": Value("xAxisTitle", "What should the title of the X-Axis be? ", "xAxisTitle [Title]", "xAxisTitle ",
		"The title of the X-Axis", "Title of X-Axis: ", lambda xAxisTitle: type(xAxisTitle) == str, lambda xAxisTitle: xAxisTitle, "", False),
	"yAxisTitle": Value("yAxisTitle", "What should the title of the Y-Axis be? ", "yAxisTitle [Title]", "yAxisTitle ",
		"The title of the Y-Axis", "Title of Y-Axis: ", lambda yAxisTitle: type(yAxisTitle) == str, lambda yAxisTitle: yAxisTitle, "", False, restrictType=["line", "scatter", "Ramachandran"]),
	"movAvg": Value("movAvg", "Would you like a moving average? (Y/N) ", "movAvg [Y | N]", "movAvg ",
		"A boolean determining if you want a moving average, useful for \"smoothing out\" noisy data.", "Moving Average? ", lambda movAvg: movAvg in ["Y", "N"], lambda movAvg: True if movAvg == "Y" else False, False, True, restrictType=["line"]),
	"movAvgFr": Value("movAvgFr", "How many frames would you like your moving average to be? ", "movAvgFr [Number of Frames]", "movAvgFr ",
		"The number of frames by which to calculate the moving average. If movAvg = False, then this is ignored.", "Moving Average Frames: ", lambda movAvgFr: movAvgFr.isdigit(), lambda movAvgFr: int(movAvgFr), 0, True, restrictType=["line"]),
	"palette": Value("palette", "Please enter the palette you would like to use, hex values separated only by a space. ", "palette [Hex Values separated by only a space]", "palette ", 
		"Hex values determining the color of each dataset. If you are using a moving average, you should input 2*numPlots, in the order: [plot1] [mov avg plot1], etc", "Palette: ", lambda palette: palette == "" or reduce(lambda s: s[0] == "#" and len(s) == 7, True, lambda x, y: x and y, palette.split(" ")), lambda palette: palette.split(" "), [], False, restrictType=["line"], advancedOption=True),
	"color": Value("color", "What are the colors/gradients are you using to color your scatter plots? You can add '_r' to reverse the order of the gradient. Separate each with a space. ", "color [New Color]", "color ", 
		"What color should each point be? See https://matplotlib.org/stable/tutorials/colors/colormaps.html for a list of colors.", "Color: ", lambda _: True, lambda color: color.split(" "), "", False, restrictType=["scatter", "Ramachandran"], advancedOption=True),
	"xLimit": Value("xLimit", "Please enter the upper and lower X-Bounds, separated by only a space. ", "xLimit [[Lower Bound] [Upper Bound]]", "xLimit ",
		"Determines the X-Axis view limits. If left > right, then the X-Axis values will decrease from left to right.", "X Bounds: ", lambda xLimit: xLimit == "" or ((len(xLimit.split(" ")) == 2 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, xLimit.split(" ")))), lambda xLimit: None if xLimit == "" else [int(x) for x in xLimit.split(" ")], [], False, advancedOption = True),
	"xTicks": Value("xTicks", "Please enter the lower bound, upper bound, and frequency of ticks, each separated by one space. ", "xTicks [[Lower Bound] [Upper Bound] [Frequency]]", "xTicks ",
		"Determines where the ticks on the X-Axis begin and end, and how frequent they are.", "X Ticks: ", lambda xTicks: xTicks == "" or ((len(xTicks.split(" ")) == 3 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, xTicks.split(" ")))), lambda xTicks: None if xTicks == "" else [int(x) for x in xTicks.split(" ")], [], False, advancedOption = True),
	"xTicksMinor": Value("xTicksMinor", "Please enter the lower bound, upper bound, and frequency of minor ticks, separated by a space. ", "xTicksMinor [[Lower Bound] [Upper Bound] [Frequency]]", "xTicksMinor ", 
		"Allows for minor ticks to be added to the plot for ease of readings.", "X Ticks (Minor): ", lambda xTicks: xTicks == "" or ((len(xTicks.split(" ")) == 3 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, xTicks.split(" ")))), lambda xTicks: None if xTicks == "" else [int(x) for x in xTicks.split(" ")], [], False, advancedOption = True),
	"xScale": Value("xScale", "What should the scale of the X-Axis be? ", "xScale [Scale]", "xScale ",
		"One unit on the X-Axis should be equal to how many units in the data?", "X Scale: ", lambda xScale: xScale == "" or IsDigit(xScale), lambda xScale: None if xScale == "" else int(xScale), 1, False, advancedOption = True),
	"xOffset": Value("xOffset", "Should the x-values be offset by any amount? ", "xOffset [Offset Value]", "xOffset ",
		"Shifts the values of the x-axis to the left or right by a specified amount", "X Offset: ", lambda xOffset: xOffset == "" or IsDigit(xOffset), lambda xOffset: None if xOffset == "" else int(xOffset), 0, False, advancedOption = True),
	"yLimit": Value("yLimit", "Please enter the upper and lower Y-Bounds, separated by only a space. ", "yLimit [[Lower Bound] [Upper Bound]]", "yLimit ",
		"Determines the Y-Axis view limits. If left > right, then the Y-Axis values will decrease from left to right.", "Y Bounds: ", lambda yLimit: yLimit == "" or ((len(yLimit.split(" ")) == 2 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, yLimit.split(" ")))), lambda yLimit: None if yLimit == "" else [int(x) for x in yLimit.split(" ")], [], False, advancedOption = True, restrictType=["line", "scatter", "Ramachandran"]),
	"yTicks": Value("yTicks", "Please enter the lower bound, upper bound, and frequency of ticks, each separated by one space. ", "yTicks [[Lower Bound] [Upper Bound] [Frequency]]", "yTicks ",
		"Determines where the ticks on the Y-Axis begin and end, and how frequent they are.", "Y Ticks: ", lambda yTicks: yTicks == "" or ((len(yTicks.split(" ")) == 3 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, yTicks.split(" ")))), lambda yTicks: None if yTicks == "" else [int(x) for x in yTicks.split(" ")], [], False, advancedOption = True, restrictType=["line", "scatter", "Ramachandran"]),
	"yTicksMinor": Value("yTicksMinor", "Please enter the lower bound, upper bound, and frequency of minor ticks, separated by a space. ", "yTicksMinor [[Lower Bound] [Upper Bound] [Frequency]]", "yTicksMinor ", 
		"Allows for minor ticks to be added to the plot for ease of readings.", "Y Ticks (Minor): ", lambda yTicks: yTicks == "" or ((len(yTicks.split(" ")) == 3 and reduce(lambda s: IsDigit(s), True, lambda x, y: x and y, yTicks.split(" ")))), lambda yTicks: None if yTicks == "" else [int(y) for y in yTicks.split(" ")], [], False, advancedOption = True),
	"yScale": Value("yScale", "What should the scale of the Y-Axis be? ", "yScale [Scale]", "yScale ",
		"One unit on the Y-Axis should be equal to how many units in the data?", "Y Scale: ", lambda yScale: yScale == "" or IsDigit(yScale), lambda yScale: None if yScale == "" else int(yScale), 1, False, advancedOption = True, restrictType=["line", "scatter", "Ramachandran"]),
	"yOffset": Value("yOffset", "Should the y-values be offset by any amount? ", "yOffset [Offset Value]", "yOffset ",
		"Shifts the values of the y-axis to the up or down by a specified amount", "Y Offset: ", lambda yOffset: yOffset == "" or IsDigit(yOffset), lambda yOffset: None if yOffset == "" else int(yOffset), 0, False, advancedOption = True, restrictType=["line", "scatter", "Ramachandran"]),
	"startNs": Value("startNs", "When calculating a series of successive nanoseconds, what number do you want to start at? ", "startNs [Nanosecond Number]", "startNs ", 
		"This option allows you to plot the Ramachandran plot of multiple nanoseconds at once. What number do you want to start at?", "StartNS: ", lambda start: IsDigit(start), lambda start: int(start), 0, False, restrictType=["Ramachandran"]),
	"endNs": Value("endNs", "When calculating a series of successive nanoseconds, what number do you want to end at (Exclusive)? ", "endNs [Nanosecond Number]", "endNs ", 
		"This option allows you to plot the Ramachandran plot of multiple nanoseconds at once. What number do you want to end at?", "EndNS: ", lambda end: IsDigit(end), lambda end: int(end), 0, False, restrictType=["Ramachandran"]),
	"indexOffset": Value("indexOffset", "Should there be any index offset? ", "indexOffset [Offset Value]", "indexOffset ", 
		"When generating one plot after another, you can keep track of the current index. Should this be offset by any amount?", "Index Offset: ", lambda offset: offset.isdigit(), lambda offset: int(offset), 0, False, True, ["Ramachandran"])
}

"""Different settings related to each individual figures the user can edit"""
FigureValues = {
	"name": Value("name", "What should the name of this figure preset be? ", "name [Name]", "name ",
	    "A name to help identify what this preset is used for.", "Name: ", lambda _: True, lambda str: str, "", True),
	"comment": Value("comment", "Optionally leave a helpful comment to yourself about the purpose of this preset.", "comment [Comment]", "comment ",
		"Just something to help you remember what this is used for when you come back to it later. Has no effect on the behavior of the preset.", "Comment: ", lambda comment: type(comment) == str, lambda comment: comment, "", False),
	"rows": Value("rows", "How many rows of sub-figures are there? ", "rows [Number of Rows]", "rows ", 
	    "Multiple sub-figures allows for more complex, less grid-based figures. How many rows of sub-figure do you want?", "Rows: ", lambda rows: rows.isdigit(), lambda rows: int(rows), 1, False, False),
	"cols": Value("cols", "How many columns of sub-figures are there? ", "cols [Number of Rows]", "cols ", 
	    "Multiple sub-figures allows for more complex, less grid-based figures. How many columns of sub-figure do you want?", "Columns: ", lambda cols: cols.isdigit(), lambda cols: int(cols), 1, False, False),
	"numSubFigures": Value("numSubFigures", "How many sub-figures do you want? ", "numSubFigures [Number of SubFigures]", "numSubFigures ",
		"How many sub-figures do you want in each subplot? For example, two rows and columns of sub-figures but only three sub-figures will delete the bottom right sub-figure.", "Number of SubFigures: ", lambda num: num.isdigit(), lambda num: str(num), False, False),
	"widthRatios": Value("widthRatios", "Please enter the ratios of widths of plots, separated by a space. ", "widthRatios [Ratio List Separated by a Space]", "widthRatios ", 
		"What should the ratio of widths be, for each plot in each row? The length of the list should be equal to the number of columns.", "Width Ratios: ", lambda wRatios: reduce(lambda num: num.isdigit(), True, lambda x, y: x and y, wRatios.split()), lambda wRatios: None if wRatios == "" else [int(x) for x in wRatios.split(" ")], [1], False, False),
	"heightRatios": Value("heightRatios", "Please enter the ratios of height of plots, separated by a space. ", "heightRatios [Ratio List Separated by a Space]", "heightRatios ", 
		"What should the ratio of height be, for each plot in each column? The length of the list should be equal to the number of rows.", "Height Ratios: ", lambda hRatios: reduce(lambda num: num.isdigit(), True, lambda x, y: x and y, hRatios.split()), lambda hRatios: None if hRatios == "" else [int(x) for x in hRatios.split(" ")], [1], False, False),
	"width": Value("width", "What should the width of the plot be? ", "width [Width]", "width ", 
		"What should the width of the figure be?", "Width: ", lambda width: width.isdigit(), lambda width: int(width), 1, True, False),
	"height": Value("height", "What should the height of the plot be? ", "height [Height]", "height ", 
		"What should the height of the figure be?", "Height: ", lambda height: height.isdigit(), lambda height: int(height), 1, True, False)
}

"""Different settings related to each individual subplots the user can edit"""
SubplotValues = {
	"name": Value("name", "What should the name of this figure preset be? ", "name [Name]", "name ",
	    "A name to help identify what this preset is used for.", "Name: ", lambda _: True, lambda str: str, "", True),
	"comment": Value("comment", "Optionally leave a helpful comment to yourself about the purpose of this preset.", "comment [Comment]", "comment ",
		"Just something to help you remember what this is used for when you come back to it later. Has no effect on the behavior of the preset.", "Comment: ", lambda comment: type(comment) == str, lambda comment: comment, "", False),
	"rows": Value("rows", "How many rows of sub-plots are there? ", "rows [Number of Rows]", "rows ", 
	    "Multiple sub-figures allows for more complex, less grid-based plots. How many rows of sub-figure do you want?", "Rows: ", lambda rows: rows.isdigit(), lambda rows: int(rows), 1, False, False),
	"cols": Value("cols", "How many columns of sub-plots are there? ", "cols [Number of Rows]", "cols ", 
	    "Multiple sub-figures allows for more complex, less grid-based plots. How many columns of sub-figure do you want?", "Columns: ", lambda cols: cols.isdigit(), lambda cols: int(cols), 1, False, False),
	"numSubPlots": Value("numSubPlots", "How many subplots do you want? ", "numSubPlots [Number of Sub-Plots]", "numSubPlots ",
		"How many sub-plots do you want in each plot? For example, two rows and columns of sub-plots but only three sub-plots will delete the bottom right sub-plot.", "Number of SubPlots: ", lambda num: num.isdigit(), lambda num: int(num), False, False),
	"widthRatios": Value("widthRatios", "Please enter the ratios of widths of plots, separated by a space. ", "widthRatios [Ratio List Separated by a Space]", "widthRatios ", 
		"What should the ratio of widths be, for each plot in each row? The length of the list should be equal to the number of columns.", "Width Ratios: ", lambda wRatios: reduce(lambda num: num.isdigit(), True, lambda x, y: x and y, wRatios.split()), lambda wRatios: None if wRatios == "" else [int(x) for x in wRatios.split(" ")], [1], False, False),
	"heightRatios": Value("heightRatios", "Please enter the ratios of height of plots, separated by a space. ", "heightRatios [Ratio List Separated by a Space]", "heightRatios ", 
		"What should the ratio of height be, for each plot in each column? The length of the list should be equal to the number of rows.", "Height Ratios: ", lambda hRatios: reduce(lambda num: num.isdigit(), True, lambda x, y: x and y, hRatios.split()), lambda hRatios: None if hRatios == "" else [int(x) for x in hRatios.split(" ")], [1], False, False),
	"shareX": Value("shareX", "Would you like each subplot to share the X-Axis? (Y/N) ", "shareX [Y | N]", "shareX ",
		"If bounds are not placed on the start and end value of the X-Axis, this will set each sub-plot to have the same start, end, etc. values.", "ShareX: ", lambda x: True if x == "Y" else False, lambda x: x, False, False),
	"shareY": Value("shareY", "Would you like each subplot to share the Y-Axis? (Y/N) ", "shareY [Y | N]", "shareY ",
		"If bounds are not placed on the start and end value of the Y-Axis, this will set each sub-plot to have the same start, end, etc. values.", "ShareY: ", lambda x: True if x == "Y" else False, lambda x: x, False, False),
}

class Dataset:
	"""An object that stores the data and relevant information from each file:
	"""
	def __init__(self, length):
		"""
		name (string): The name assigned to this set of data,
		length (int): The number of rows in the file, used to initialize the list to avoid repeatedly appending,
		data (int list list): The actual data stored in the file, Data[n][0]: row number, Data[n][1]: value associated with its row.
		movAvg (int list): If the user wants a moving average, stores the value of that average, starting at index "amount"-1, leaving those indices None.
		amount (int): The number of frames for the moving average.
		"""
		self.name, self.path = None, None
		self.length = length
		self.data = []
		self.movAvg = []

	def Prompt(self):
		"""
		Gets the name and path to a dataset
		"""
		self.name = input("What should the name of this data be?\n--- ")
		self.path = input("What\'s the path to " + self.name + "?\n--- ")
		while not os.path.isfile(self.path):
			print("That is not a valid file.")
			self.path = input("What\'s the path to " + self.name + "?\n--- ")

	def Populate(self):
		"""
		Reads in the provided path and stores all the data in self.data
		"""
		
		if self.path[len(self.path)-4:] == ".tsv":
			with open(self.path) as file:
				rd = csv.reader(file, delimiter="\t")
				for index, value in enumerate(rd):
					if index != 0:
						self.data[index] = [float(x) for x in value[1:]]
		elif self.path[len(self.path)-4:] == ".csv":
			with open(self.path) as file:
				rd = csv.reader(file, delimiter=",")
				for index, value in enumerate(rd):
					# self.data[index] = [float(x) for x in value[1:]]
					if index != 0:
						self.data.append([float(x) for x in value[1:]])
		elif self.path[len(self.path)-4:] == ".dat":
			print(f"Begin Reading {self.path}")
			self.data = [0] * self.length
			with open(self.path) as fd:
				for lineNumber, line in enumerate(fd):
					if lineNumber != 0: #We don't want a header
						#The file should be in the format: "   ###1      ###2", where ###1 is the index, and ###2 is the value
						#The following line will, for each line, give the following list: ["   ###1", "      ###2"]
						l = re.findall(" *-?\d+\.?\d*", line)
						modified = [0] * len(l)
						for index, value in enumerate(l):
							modified[index] = float(value.replace(" ", ""))
						#modified will now equal [###1, ###2]
						if modified != []:
							# modified[0] = float(modified[0])
							self.data[lineNumber - 1] = modified
			#self.data is now a list of lists, but we're only interested in the second value of each list
			#if self.length is larger than the number of rows in the file, then there will be trailing zeros after the data,
			#the inner loop comprehension removes them so that the zip doesn't throw an error
			# print(self.data)
			self.data = [[x[0], x[1:]] if type(x) == list else x for x in self.data]
			# self.data = [list(i) for i in zip(*[x for x in self.data if x != 0])]
			# print(self.data)
			# for x in self.data:
			# 	print(x)
			self.data = [x[1] if type(x) == list else [x] for x in self.data]
			# print(self.data)
			print(f"End Reading {self.path}")
		else:
			raise ValueError("Data file must be in .csv, .tsv, or .dat format.")

	def CalcMovingAvg(self, amount):
		"""
		Calculates a moving average, given amount frames
		"""
		print(f"Begin Moving Average {self.path}")
		# self.movAvg = [None] * self.length
		self.movAvg = [None] * len(self.data)
		self.amount = amount
		# for x in range(amount-1, self.length): #999 to 1001000
		for x in range(amount-1, len(self.data)):
			avg = 0
			# print(f"Inner bound: {x - (amount-1)} to {x+1}, len movAvg: {len(self.movAvg)}")
			for y in range(x - (amount-1), x+1): #999-999=0 to 1000
				try:
					avg += self.data[y]
				except IndexError:
					break;
				except:
					print("There's a problem with your file.")
			avg /= amount
			self.movAvg[x] = avg
		print(f"End Moving Average {self.path}")

	def AsDataFrame(self):
		"""
		Converts the Dataset to a DataFrame format
		"""
		if self.movAvg:
			print(f"Len data: {len(self.data)}, len avg: {len(self.movAvg)}")
			dataFrame = pd.DataFrame({self.name : self.data, f"{self.name} {self.amount} Frame\nMov Avg" : self.movAvg})
		else:
			dataFrame = pd.DataFrame({self.name : self.data})
		return dataFrame

class Preset:
	def asString(self, showAdvanced) -> str:
		"""Returns the Preset as a string of its values

		Args:
			showAdvanced (bool): 

		Returns:
			str: String of Preset values, each on a new line with title
		"""
		toReturn = ""
		for x in list(self.values.values()):
			if not (x.advancedOption == True and showAdvanced == False):
				if "type" in list(map(lambda x: x.name, list(self.values.values()))):
					if self.values["type"].value in x.restrictType or x.restrictType == []:
						toReturn += f"{x.display}{x.value}\n"
				else:
					toReturn += f"{x.display}{x.value}\n"
		if showAdvanced == False and reduce(lambda x: x.advancedOption, False, lambda x, y: x or y, list(self.values.values())):
			toReturn += "---Advanced Options Hidden---"
		return toReturn
	
	def getValues(self) -> dict[str, any]:
		"""Returns a dictionary containing the name of each value, and the value associated with it

		Returns:
			dict[str, any]: A dictionary containing the name and value associated with each value in the Preset
		"""
		return {n.name: n.value for n in list(self.values.values())}

	def copy(self, Preset):
		"""Returns a copy of itself

		Returns:
			Preset: Copy of itself
		"""
		#Takes the name, Value dictionary of values (self.values), extracts the values and converts it to a list of Values.
		#Then uses list comprehension to extract the value attribute from each Value object.
		#Finally, unpacks the list and passes it into the Preset constructor
		return Preset(*[x.value for x in list(self.values.values())])

class AxisPreset(Preset):
	"""
	An established group of settings by which to graph data.
	"""
	def __init__(self, name: str = "", comment: str = "", type: str = "", numRows: int = 0, numPlots: int = 0, xAxisTitle: str = "", yAxisTitle: str = "", movAvg: bool = False, 
	    	movAvgFr: int = 0, palette: list[str] = [], color: list[str] = [], xLimit: list[int] = [], xTicks: list[str] = [], xTicksMinor: list[int] = [], xScale: int = 1, xOffset: int = 0,
			yLimit: int = 0, yTicks: list[int] = [], yTicksMinor: list[int] = [], yScale: int = 0, yOffset: int = 0, startNs: int = 0, endNs: int = 0, indexOffset : int = 0):
		"""_summary_

		Args:
			name (str, optional): Name of the Preset. Defaults to "".
			comment (str, optional): A helpful comment to provide more context preset. Defaults to "".
			type (str, optional): The type or style of plot the user wants to create. Defaults to "".
			numRows (int, optional): Of all the files that contain data to visualize, what is the maximum number of rows in them? Defaults to 0.
			numPlots (int, optional): The number of datasets to plot. Defaults to 0.
			xAxisTitle (str, optional): The title of the X-Axis. Defaults to "".
			yAxisTitle (str, optional): The title of the Y-Axis. Defaults to "".
			movAvg (bool, optional): Is there a moving average on this dataset? Defaults to False.
			movAvgFr (int, optional): If there is a moving average, the number of frames. Defaults to 0.
			palette (list[str], optional): A list of hex values corresponding to the color of plots. Defaults to [].
			color (list[str], optional): The colors/gradients if the user is graphing a scatter plot. Defaults to [].
			xLimit (list[int], optional): The maximum value shown on the X-Axis. Defaults to [].
			xTicks (list[str], optional): The start, end, and frequency of tick marks on the X-Axis. Defaults to [].
			xTicksMinor (list[int], optional): The start, end, and frequency of minor ticks on the X-Axis. Defaults to [].
			xScale (int, optional): Scales the values on the X-Axis by a factor. Defaults to 1.
			xOffset (int, optional): Adds an offset to each of the values. Defaults to 0.
			yLimit (int, optional): The maximum value shown on the Y-Axis. Defaults to 0.
			yTicks (list[int], optional): The start, end, and frequency of tick marks on the Y-Axis. Defaults to [].
			yTicksMinor (list[int], optional): The start, end and frequency of minor ticks on the Y-Axis. Defaults to [].
			yScale (int, optional): Scales the values on the Y-Axis by a factor. Defaults to 0.
			yOffset (int, optional): Adds an offset to each of the values. Defaults to 0.
			startNs (int, optional): If the user is plotting a range of nanoseconds, what is the starting value? Defaults to 0.
			endNs (int, optional): If the user is plotting a range of nanoseconds, what is the end value? Defaults to 0.
			indexOffset (int, optional): Adds an offset to the nanosecond count. Defaults to 0.
		"""
		self.values = copy.deepcopy(AxisValues)
		self.values["name"].value = name
		self.values["comment"].value = comment
		self.values["type"].value = type
		self.values["numRows"].value = numRows
		self.values["numPlots"].value = numPlots
		self.values["xAxisTitle"].value = xAxisTitle
		self.values["yAxisTitle"].value = yAxisTitle
		self.values["movAvg"].value = movAvg
		self.values["movAvgFr"].value = movAvgFr
		self.values["palette"].value = palette
		self.values["color"].value = color
		self.values["xLimit"].value = xLimit
		self.values["xTicks"].value = xTicks
		self.values["xTicksMinor"].value = xTicksMinor
		self.values["xScale"].value = xScale
		self.values["xOffset"].value = xOffset
		self.values["yLimit"].value = yLimit
		self.values["yTicks"].value = yTicks
		self.values["yTicksMinor"].value = yTicksMinor
		self.values["yScale"].value = yScale
		self.values["yOffset"].value = yOffset
		self.values["startNs"].value = startNs
		self.values["endNs"].value = endNs
		self.values["indexOffset"].value = indexOffset
	

class FigurePreset(Preset):
	def __init__(self, name : str = "", comment : str = "", rows : int = 0, cols : int = 0, numSubFigures : int = -1, widthRatios : list[int] = [], 
	    	heightRatios: list[int] = [], width : int = 0, height : int = 0) -> None:
		"""
		Args:
			name (str, optional): The name of this group of settings. Defaults to "".
			comment (str, optional): A helpful comment to provide more context preset. Defaults to "".
			rows (int, optional): The number of rows of sub-figures. Defaults to 0.
			cols (int, optional): The number of columns of sub-figures. Defaults to 0.
			numSubFigures (int, optional): The number of sub-figures. Defaults to -1.
			widthRatios (list[int], optional): The ratio of widths of rows of sub-figures. Defaults to [].
			heightRatios (list[int], optional): The ratio of hights of rows of sub-figures. Defaults to [].
			width (int, optional): The width of the image. Defaults to 0.
			height (int, optional): The hight of the image. Defaults to 0.
		"""
		self.values = copy.deepcopy(FigureValues)
		self.values["name"].value = name
		self.values["comment"].value = comment
		self.values["rows"].value = rows
		self.values["cols"].value = cols
		self.values["numSubFigures"].value = rows * cols if numSubFigures < 0 else numSubFigures
		self.values["widthRatios"].value = widthRatios
		self.values["heightRatios"].value = heightRatios
		self.values["width"].value = width
		self.values["height"].value = height

class SubplotPreset(Preset):
	def __init__(self, name: str, comment : str = "", rows : int = 0, cols : int = 0, numSubPlots : int = -1, widthRatios : list[int] = [], heightRatios : list[int] = [],
	    	shareX : bool = False, shareY : bool = False) -> None:
		"""
		Args:
			name (str): The name of this group of settings. 
			comment (str, optional): A helpful comment to provide more context preset. Defaults to "".
			rows (int, optional): The number of rows of subplots in this figure. Defaults to 0.
			cols (int, optional): The number of columns of subplots in this figure. Defaults to 0.
			numSubPlots (int, optional): The number of subplots. Defaults to -1.
			widthRatios (list[int], optional): The ratio of widths of subplots. Defaults to [].
			heightRatios (list[int], optional): The ratio of heights of subplots. Defaults to [].
			shareX (bool, optional): Should the subplots share the same X-Axis? Defaults to False.
			shareY (bool, optional): Should the subplots share the same Y-Axis? Defaults to False.
		"""
		self.values = copy.deepcopy(SubplotValues)
		self.values["name"].value = name
		self.values["comment"].value = comment
		self.values["rows"].value = rows
		self.values["numSubPlots"].value = numSubPlots
		self.values["cols"].value = cols
		self.values["widthRatios"].value = widthRatios
		self.values["heightRatios"].value = heightRatios
		self.values["shareX"].value = shareX
		self.values["shareY"].value = shareY

def new(Presets, presetFile, presetType):
	"""
	Handles when the user wants to create a new Preset.

	Returns the newly created Preset.
	"""
	#[x.value for x in list(Values.values())] is the list of all default values for each Value object
	Preset = None
	if presetType == FigurePreset:
		Preset = presetType(*[x.value for x in list(FigureValues.values())])
	elif presetType == SubplotPreset:
		Preset = presetType(*[x.value for x in list(SubplotValues.values())])
	else:
		Preset = presetType(**{x.name: x.value for x in list(AxisValues.values())})

	print("Note:\n\
If you choose to save your file and you have not chosen a unique name, it will override the original.\nQuestions preceded by an asterisk (*) are mandatory, leaving optional questions will use the default settings. Type 'Help' for more information about any of the options.")

	#Get user input for each of the values
	for x in list(Preset.values.values()):
		prompt = ""
		if x.restrictType == [] or Preset.values["type"].value in x.restrictType:
			if x.isMandatory == True:
				prompt = f"*{x.newPrompt}"
			else:
				prompt = x.newPrompt
			userIn = input(f"{prompt}\n--- ")
			helpMessage = False
			if userIn in ["Help", "help"]:
				print(x.help)
				helpMessage = True
			while (x.isMandatory == True and userIn == "") or (not x.verify(userIn)) or helpMessage == True:
				if helpMessage != True: print("Invalid input.")
				userIn = input(f"{prompt}\n--- ")
			x.value = x.convert(userIn)

	save = input("Would you like to save this preset? (Y/N)\n--- ")
	while save not in ["Y", "N"]:
		print("Invalid input.")
		save = input("Would you like to save this preset? (Y/N)\n--- ")

	if save == "Y":
		#converts this preset to a dictionary
		dict = {Preset.values["name"].value: Preset.getValues()}
		#Converts all the other presets to dictionaries and combines them all together
		for x in list(Presets.values()):
			dict |= {x.values["name"].value: x.getValues()}

		with open(presetFile, "w", encoding="utf-8") as f:
			json.dump(dict, f, ensure_ascii=False, indent=4)

	return Preset

def modify(Presets, presetFile, presetType) -> AxisPreset:
	"""
	Handles when the user wants to modify an existing Preset.

	Returns the modified Preset. Does not modify original Preset.
	"""
	modify = input("Which preset would you like to modify?\n--- ")
	while modify not in Presets:
		print("Invalid input.")
		modify = input("\n--- ")

	#Need to create a copy otherwise all the values will just be overwritten
	oldPreset = Presets[modify].copy(presetType)

	print("You can use the following commands to modify the preset:")
	for x in list(oldPreset.values.values()):
		if x.restrictType == [] or oldPreset.values["type"].value in x.restrictType:
			print(x.modifyPrompt)

	print("Type \"done\" when you are done.\n\
If you choose to save your file and you have not chosen a unique name, it will override the original.\n\
Otherwise it will save as a copy of the original.")
       
	print("Type \"Help [Command Name]\" to get more info about a specific command.")

	#Get user input to change values of preset
	command = input("\n--- ")
	while(command != "done"):
		if command[:4] in ["Help", "help"]:
			if command[5:] in list(oldPreset.values.keys()):
				print(oldPreset.values[command[5:]].help)
			else:
				print("Invalid input.")
		else:
			for x in list(oldPreset.values.keys()):
				if command[:len(oldPreset.values[x].modifyCommand)] == oldPreset.values[x].modifyCommand:
					if oldPreset.values[x].verify(command[len(oldPreset.values[x].modifyCommand):]):
						oldPreset.values[x].value = oldPreset.values[x].convert(command[len(oldPreset.values[x].modifyCommand):])
					else:
						print("Invalid input.")

		command = input("\n--- ")

	save = input("Would you like to save this preset? (Y/N)\n--- ")
	while save not in ["Y", "N"]:
		print("Invalid input.")
		save = input("Would you like to save this preset? (Y/N)\n--- ")

	if save == "Y":
		#Converts this preset to a dictionary
		dict = {}
		for x in list(Presets.values()):
			#Converts each other preset to a dictionary and combines them all together
			dict |= {x.values["name"].value: x.getValues()}
		dict |= {oldPreset.values["name"].value: oldPreset.getValues()}

		with open(presetFile, "w", encoding="utf-8") as f:
			json.dump(dict, f, ensure_ascii=False, indent=4)

	return oldPreset

def greeting(Presets, presetFile) -> list[str]:
	"""
	Greets the user, displays any loaded Presets, and asks what they would like to do
	"""
	keys = list(Presets.keys())

	if keys != []:
		print("Here are your loaded presets:\n")
		for preset in keys:
			print(Presets[preset].asString(False))
			print("")
		print("Please enter the name of the preset you would like to use.\n\
Enter \"new\" if you would like to create a new preset from scratch,\n\
or enter \"modify\" if you would like to edit an existing preset.")
		if type(list(Presets.values())[0]) == AxisPreset:
			print("Finally, you may type \"advanced [preset name]\" to view all the settings associated with that preset.")
		print("You will be prompted to enter the title of the picture, the title of the graph, and the subtitle later.")
	else:
		print(f"You have no presets on file. If you believe you should, check to find a {presetFile} file in this directory.")
		print("Enter \"new\" to create a new preset from scratch.")

	keys.append("new")
	keys.append("modify")

	for x in list(Presets.keys()):
		keys.append("advanced " + Presets[x].values["name"].value)

	return keys

def getPreset(Presets, presetFile, keys, presetType) -> AxisPreset:
	"""
	Determines which AxisPreset they would like to use, if they would like to create a new AxisPreset, or modify an existing AxisPreset.

	Returns the AxisPreset they have chosen to use.

	Args:
		AxisPresets (AxisPreset dict): The user's loaded AxisPresets
		presetFile (string): a string locating the user's file or presets
		keys (string list): a list of strings of acceptable user input
	"""

	preset = input("\n--- ")
	while preset not in keys or (preset == "modify" and Presets == {}) or (preset[:8] == "advanced" and Presets == {}):
		print("Invalid input.")
		print(preset[:8] == "advanced" and Presets == {})
		preset = input("\n--- ")

	if preset == "new":
		preset = new(Presets, presetFile, presetType)
	elif preset == "modify":
		preset = modify(Presets, presetFile, presetType)
	elif preset[:8] == "advanced":
		print(Presets[preset[9:]].asString(True))
		return getPreset(Presets, presetFile, keys, presetType)
	else:
		preset = Presets[preset]

	return preset


def getVariableValues(numSubplots: str) -> tuple[str, str, list[str], list[str]]:
	"""Gets input from user that are likely to vary each time the script is run

	Returns:
		tuple[str, str, str]: The name of the picture, the title of the graph, the subtitle of the graph
	"""
	pictureName = input("What should the picture of the plot be named?\n--- ")
	title = input("What should the title of the graph be?\n--- ")
	# suptitle = input("What should the subtitle of the graph be? ")
	subfigureTitles = []
	subplotTitles = []
	print("Starting from top left to bottom right, what are the titles of each subplot?")
	for index, value in enumerate(numSubplots):
		subfigureTitles.append(input(f"What should the name of figure {index + 1} be?\n--- "))
		temp = [""] * value
		for j in range(value):
			temp[j] = input(f"Title of graph {j+1}:\n--- ")
		subplotTitles.append(temp)
	return (pictureName, title, subfigureTitles, subplotTitles)

def populateDatasets(axisPresetList: list[list[AxisPreset]], numSubPlots: list[int]) -> pd.DataFrame:
	"""Populates each dataset by getting input from the user

	Args:
		axisPresetList (list[list[AxisPreset]]): List of all the presets loaded

	Returns:
		pd.DataFrame: Dataframe containing all the data for the graph
	"""

	dataPlots = []
	for index, i in enumerate(numSubPlots):
		temp = [pd.DataFrame({})] * i
		for j in range(i):
			plots = []
			print(axisPresetList[index][j].values["name"].value, axisPresetList[index][j].values["numPlots"].value)
			for x in range(axisPresetList[index][j].values["numPlots"].value):
				x = Dataset(axisPresetList[index][j].values["numRows"].value)
				x.Prompt()
				plots.append(x)

			for x in plots:
				x.Populate()
				temp[j] = pd.concat([temp[j], x.AsDataFrame()], axis = 1)
				if axisPresetList[index][j].values["movAvg"].value:
					temp[j][f"{x.name} " + str(axisPresetList[index][j].values["movAvgFr"].value) + " Frame Mov. Avg."] = temp[j][x.name].apply(lambda x: x[0]).rolling(window = axisPresetList[index][j].values["movAvgFr"].value).mean()

			temp[j].reset_index(inplace=True)
			xScale = axisPresetList[index][j].values["xScale"].value
			yScale = axisPresetList[index][j].values["yScale"].value
			xOffset = axisPresetList[index][j].values["xOffset"].value
			yOffset = axisPresetList[index][j].values["yOffset"].value

			if xScale == None:
				xScale = 1
			if yScale == None:
				yScale = 1
			if xOffset == None:
				xOffset = 0
			if yOffset == None:
				yOffset = 0

			temp[j]["index"] = temp[j]["index"].apply(lambda x: x/xScale + xOffset)
			for x in plots:
				if type(temp[j][x.name].iloc[0]) == list:
					# print(temp[j][x.name])
					temp[j][x.name] = temp[j][x.name].apply(lambda y: [item/yScale + yOffset for item in y] if type(y) == list else y)
				else:
					temp[j][x.name] = temp[j][x.name].apply(lambda y: y/yScale + yOffset)

			temp[j] = pd.melt(temp[j], ["index"])
		dataPlots.append(temp)
	return dataPlots

def loadPresets(Presets: dict[str, Preset], presetFile: str, presetType) -> dict[str, AxisPreset]:
	"""Reads in any presets from presetFile and appends them to a dictionary.

	If the file does not exist, function throws a warning to the user, and returns an empty dictionary.

	Args:
		Presets (dict[str, Preset]): An empty dictionary
		presetFile (str): A string to where the Preset JSON file can be located
		presetType: the type of preset to load

	Returns:
		dict[str, Preset]: The dictionary containing the name, and the associated Preset. Empty if Preset file does not exist.
	"""
	data = {}
	if os.path.isfile(presetFile):
		with open(presetFile) as file:
			data = json.load(file)
		for x in list(data.keys()):
			#currDict will "collect" each name, value of the current preset
			currDict = {}
			for y in data[x]:
				currDict |= {y: data[x][y]}
			#To pass each required argument into the Preset class, the values of the currDict dictionary are converted to a dictionary, then unpacked into the constructor function
			# Presets |= {x: presetType(*list(currDict.values()))}
			# print(list(currDict.values()))
			# print({x: x for x in list(currDict.values())})
			Presets |= {x: presetType(**currDict)}

	return Presets

# https://stackoverflow.com/a/56253636/13351405
def legend_without_duplicate_labels(axs):
	all_unique = []
	for ax in axs:
		handles, labels = ax.get_legend_handles_labels()
		# print(handles, labels)
		unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
		# print(f"Handles: {handles}, labels: {labels}")
		# print(f"Unique: {unique}")
		for (h, l) in unique:
			# print(u, unique)
			if l not in [y for (_, y) in all_unique]:
				all_unique.append((h, l))
	return zip(*all_unique)

def plot(axisPreset, subplotPreset, figurePreset, pictureName, title, suptitle, dataPlots, subplotTitles):
	# plt.figure(figsize = (6, 4))
	fig, axs = plt.subplots(1, 2, figsize = (12, 12), sharex=True, sharey=True)
	axs = axs.flatten()
	sns.set(font = "Helvetica")
	plt.suptitle(title, ha="center", x = 0.515)
	if suptitle != "":
		plt.subplots_adjust(top=0.8, wspace=0.1, hspace=0.1)
		plt.text(x = 0.515, y = 0.87, s = suptitle, fontsize=12, ha="center", c="#4b4b4b", transform=fig.transFigure)
	else:
		plt.subplots_adjust(wspace=0.1, hspace=0.1)
	for index, axes in enumerate(axs):
		print(f"Working on plot {index + 1}")
		if axisPreset.values["palette"].value == [""]:
			ax = sns.lineplot(data = dataPlots[index], x = "index", y = "value", dashes = False, hue = "variable", ax = axes)
			ax.legend_.remove()
		else:
			ax = sns.lineplot(data = dataPlots[index], x = "index", y = "value", dashes = False, hue = "variable", palette = axisPreset.values["palette"].value, ax = axes)
			ax.legend_.remove()
		axes.set(xlabel = axisPreset.values["xAxisTitle"].value, ylabel = axisPreset.values["yAxisTitle"].value)
		if subplotTitles != []:
			axes.set_title(subplotTitles[index], fontsize="small", alpha=0.8, ha="center", x = 0.5)
		# axes.legend(borderaxespad = 0)
		# sns.move_legend(ax, bbox_to_anchor = (.5, -.15 ), ncol = 3, loc = "upper center", edgecolor = "white")
		axes.grid(visible = True, color = "#D9D9D9")
		if axisPreset.values["xLimit"].value != None:
			ax.set_xlim(axisPreset.values["xLimit"].value[0], axisPreset.values["xLimit"].value[1])
		if axisPreset.values["xTicks"].value != None:
			ax.set_xticks(range(axisPreset.values["xTicks"].value[0], axisPreset.values["xTicks"].value[1], axisPreset.values["xTicks"].value[2]))
		if axisPreset.values["yLimit"].value != None:
			ax.set_ylim(axisPreset.values["yLimit"].value[0], axisPreset.values["yLimit"].value[1])
		if axisPreset.values["yTicks"].value != None:
			ax.set_xticks(range(axisPreset.values["yTicks"].value[0], axisPreset.values["yTicks"].value[1], axisPreset.values["yTicks"].value[2]))
		axes.ticklabel_format(style='plain')
		axes.margins(0)
	fig.legend(*legend_without_duplicate_labels(axs), borderaxespad = 0, bbox_to_anchor = (0.5, 0), ncol = 3, loc = "upper center", edgecolor = "white", framealpha = 0)
	fig.savefig(pictureName, bbox_inches = "tight", dpi = 500)

def linePlot(axisPreset, ax, title, dataPlots):
	ax.set_title(title)
	# print(f"Working on plot {index + 1}")
	print(dataPlots)
	dataPlots["value"] = dataPlots["value"].apply(lambda x: x[0] if type(x) == list else x)
	print(dataPlots)
	if axisPreset.values["palette"].value == [""]:
		ax = sns.lineplot(data = dataPlots, x = "index", y = "value", dashes = False, hue = "variable", ax = ax)
		ax.legend_.remove()
	else:
		ax = sns.lineplot(data = dataPlots, x = "index", y = "value", dashes = False, hue = "variable", palette = axisPreset.values["palette"].value, ax = ax)
		ax.legend_.remove()
	ax.grid(visible = True, color = "#D9D9D9")
	ax.set(xlabel = axisPreset.values["xAxisTitle"].value, ylabel = axisPreset.values["yAxisTitle"].value)
	if axisPreset.values["xLimit"].value != None:
		ax.set_xlim(axisPreset.values["xLimit"].value[0], axisPreset.values["xLimit"].value[1])
	if axisPreset.values["xTicks"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicks"].value[0], axisPreset.values["xTicks"].value[1], axisPreset.values["xTicks"].value[2]))
	if axisPreset.values["yLimit"].value != None:
		ax.set_ylim(axisPreset.values["yLimit"].value[0], axisPreset.values["yLimit"].value[1])
	if axisPreset.values["yTicks"].value != None:
		ax.set_xticks(range(axisPreset.values["yTicks"].value[0], axisPreset.values["yTicks"].value[1], axisPreset.values["yTicks"].value[2]))
	if axisPreset.values["xTicksMinor"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicksMinor"].value[0], axisPreset.values["xTicksMinor"].value[1], axisPreset.values["xTicks"].value[2]), minor=True)
	if axisPreset.values["yTicksMinor"].value != None:
		ax.set_xticks(range(axisPreset.values["yTicksMinor"].value[0], axisPreset.values["yTicksMinor"].value[1], axisPreset.values["yTicks"].value[2]), minor=True)
	ax.ticklabel_format(style='plain')
	ax.margins(0)

def scatterPlot(axisPreset, ax, title, dataPlots):
	ax.set_title(title)
	for index in range(len(dataPlots.iloc[0]["value"])):
		# idk = ax.scatter(x=dataPlots["index"], y=dataPlots["value"].apply(lambda x: x[index]), cmap=axisPreset.values["color"].value[0], c=list(range(len(dataPlots["index"]) + 1, 1, -1)))
		ax.scatter(x=dataPlots["index"], y=dataPlots["value"].apply(lambda x: x[index]), cmap=axisPreset.values["color"].value[0], c=list(range(len(dataPlots["index"]) + 1, 1, -1)))
	ax.set_xlabel(axisPreset.values["xAxisTitle"].value)
	ax.set_ylabel(axisPreset.values["yAxisTitle"].value)
	if axisPreset.values["xLimit"].value != None:
		ax.set_xbound(lower = axisPreset.values["xLimit"].value[0], upper = axisPreset.values["xLimit"].value[1])
	if axisPreset.values["yLimit"].value != None:
		ax.set_ybound(lower = axisPreset.values["yLimit"].value[0], upper = axisPreset.values["yLimit"].value[1])
	if axisPreset.values["xTicks"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicks"].value[0], axisPreset.values["xTicks"].value[1], axisPreset.values["xTicks"].value[2]))
	if axisPreset.values["yTicks"].value != None:
		ax.set_yticks(range(axisPreset.values["yTicks"].value[0], axisPreset.values["yTicks"].value[1], axisPreset.values["yTicks"].value[2]))
	if axisPreset.values["xTicksMinor"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicksMinor"].value[0], axisPreset.values["xTicksMinor"].value[1], axisPreset.values["xTicks"].value[2]), minor=True)
	if axisPreset.values["yTicksMinor"].value != None:
		ax.set_yticks(range(axisPreset.values["yTicksMinor"].value[0], axisPreset.values["yTicksMinor"].value[1], axisPreset.values["yTicks"].value[2]), minor=True)

	ax.tick_params(axis="both", which = "minor")
	ax.grid(visible=True, which="minor", axis="both")
	# ax.grid(visible=True, which="major", axis="both", color="black", lw=2)
	ax.grid(visible=True, which="major", axis="both")
	ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
	ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
	ax.set_axisbelow(True)
	# ax.legend([idk, (idk)], ["data"], scatterpoints = 3)

def ramachandranPlot(axisPreset, ax, title, dataPlots, ns):
	ax.set_title(title)
	for index, variable in enumerate(list(dict.fromkeys(list(dataPlots["variable"])))):
		currVariable = np.where(dataPlots["variable"] == variable)
		x = dataPlots.iloc[currVariable[0]]["value"].apply(lambda x: x[2 * ns] if type(x) == list else x)
		y = dataPlots.iloc[currVariable[0]]["value"].apply(lambda x: x[2 * ns + 1] if type(x) == list else x)
		ax.scatter(x=x, y=y, cmap=axisPreset.values["color"].value[index], c=list(range(len(dataPlots.iloc[currVariable[0]]["value"]) + 1, 1, -1)))
	ax.set_xlabel(axisPreset.values["xAxisTitle"].value)
	ax.set_ylabel(axisPreset.values["yAxisTitle"].value)
	if axisPreset.values["xLimit"].value != None:
		ax.set_xbound(lower = axisPreset.values["xLimit"].value[0], upper = axisPreset.values["xLimit"].value[1])
	if axisPreset.values["yLimit"].value != None:
		ax.set_ybound(lower = axisPreset.values["yLimit"].value[0], upper = axisPreset.values["yLimit"].value[1])
	if axisPreset.values["xTicks"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicks"].value[0], axisPreset.values["xTicks"].value[1], axisPreset.values["xTicks"].value[2]))
	if axisPreset.values["yTicks"].value != None:
		ax.set_yticks(range(axisPreset.values["yTicks"].value[0], axisPreset.values["yTicks"].value[1], axisPreset.values["yTicks"].value[2]))
	if axisPreset.values["xTicksMinor"].value != None:
		ax.set_xticks(range(axisPreset.values["xTicksMinor"].value[0], axisPreset.values["xTicksMinor"].value[1], axisPreset.values["xTicksMinor"].value[2]), minor=True)
	if axisPreset.values["yTicksMinor"].value != None:
		ax.set_yticks(range(axisPreset.values["yTicksMinor"].value[0], axisPreset.values["yTicksMinor"].value[1], axisPreset.values["yTicksMinor"].value[2]), minor=True)

	ax.tick_params(axis="both", which = "minor")
	ax.grid(visible=True, which="minor", axis="both")
	ax.grid(visible=True, which="major", axis="both", color="black", lw=2)
	# ax.grid(visible=True, which="major", axis="both")
	ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
	ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
	ax.set_axisbelow(True)
	# ax.legend([idk, (idk)], ["data"], scatterpoints = 3)

def gradientPlot(axisPreset, ax):
	gradient = np.linspace(1000, 2, 1000)
	gradient = np.vstack((gradient, gradient))
	ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[axisPreset.values["color"].value[0]])
	ax.axes.get_yaxis().set_visible(False)
	ax.axes.set_xlim(left=2)
	ax.axes.set_xticks([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
	ax.axes.set_xlabel(axisPreset.values["xAxisTitle"].value)

def getStartEndOffset(axisPresetList):
	start = 0
	end = 1
	for i in axisPresetList:
		for j in i:
			if j.values["type"].value == "Ramachandran":
				start = j.values["startNs"].value
				end = j.values["endNs"].value
				indexOffset = j.values["indexOffset"].value
	return (start, end, indexOffset)

def plotHelper(axisPresetList, subplotPresetList, figurePreset, pictureName, title, dataPlots, subfigureTitles, subplotTitles):
	start = 0
	end = 1
	offset = 0
	flag = "!#!#"
	# if any of the types of plots to be generated are Ramachandran
	if reduce(lambda x: True if True in [y.values["type"].value == "Ramachandran" for y in x] else False, False, lambda x, y: x or y, axisPresetList):
		start, end, offset = getStartEndOffset(axisPresetList)
		print(offset)
	
	# loops though each of the nanoseconds, if the plot isn't Ramachandran this loop just executes once
	for ns in range(start, end):
		fig = plt.figure(layout='constrained', figsize=(figurePreset.values["width"].value, figurePreset.values["height"].value))
		fig.subplots_adjust(hspace=0.4, top=0.925, bottom=0.1)
		fig.suptitle(title.replace(flag, str(ns)), fontsize=20)
		
		subfigs = fig.subfigures(figurePreset.values["rows"].value, figurePreset.values["cols"].value, 
				height_ratios=figurePreset.values["heightRatios"].value, width_ratios=figurePreset.values["widthRatios"].value, 
				hspace = 0.01, wspace = 0.01)

		# The fig.subfigures method is very annoying; sometimes it's type np.ndarray sometimes it isn't
		if type(subfigs) != np.ndarray:
			subfigs = np.ndarray(shape=(1,), buffer=np.array([subfigs]), dtype=mpl.figure.SubFigure)

		# Loops though all the subfigures
		for i, subfig in enumerate(subfigs):
			if i+1 <= figurePreset.values["numSubFigures"].value:
				subfig.suptitle(subfigureTitles[i].replace(flag, str(ns)))
				axes = subfig.subplots(subplotPresetList[i].values["rows"].value, subplotPresetList[i].values["cols"].value, 
					height_ratios = subplotPresetList[i].values["heightRatios"].value, width_ratios = subplotPresetList[i].values["widthRatios"].value,
					sharex = subplotPresetList[i].values["shareX"].value, sharey = subplotPresetList[i].values["shareY"].value)
				
				if type(axes) == np.ndarray:
					axes = axes.flatten()
				else:
					axes = np.ndarray(shape=(1,), dtype=mpl.axes.Axes, buffer=np.array([axes]))
				
				# Loops though all the subplots
				for j, ax in enumerate(axes):
					if j+1 <= subplotPresetList[i].values["numSubPlots"].value:
						print(f"Working on plot {j + 1}")
						match axisPresetList[i][j].values["type"].value:
							case "line":
								linePlot(axisPresetList[i][j], ax, subplotTitles[i][j], dataPlots[i][j])
							case "scatter":
								scatterPlot(axisPresetList[i][j], ax, subplotTitles[i][j], dataPlots[i][j])
							case "Ramachandran":
								ramachandranPlot(axisPresetList[i][j], ax, subplotTitles[i][j], dataPlots[i][j], ns - offset)
							case "gradient":
								gradientPlot(axisPresetList[i][j], ax)
					else:
						subfig.delaxes(ax)
				subfig.legend(*legend_without_duplicate_labels(axes), borderaxespad = 0, bbox_to_anchor = (0.5, 0), ncol = 3, loc = "upper center", edgecolor = "white", framealpha = 0, scatterpoints=3)
			else:
				#remove subfigure
				pass

		fig.savefig(pictureName.replace(flag, str(ns)), bbox_inches = "tight", dpi = 500)
			

def main():
	sns.set_theme(style="ticks")
	figurePresetFile = "presets_figure.json"
	subplotPresetFile = "presets_subplot.json"
	axisPresetFile = "presets_axis.json"
	FigurePresets = {}
	SubplotPresets = {}
	AxisPresets = {}

	FigurePresets = loadPresets(FigurePresets, figurePresetFile, FigurePreset)
	SubplotPresets = loadPresets(SubplotPresets, subplotPresetFile, SubplotPreset)
	AxisPresets = loadPresets(AxisPresets, axisPresetFile, AxisPreset)

	keys = greeting(FigurePresets, figurePresetFile)
	figurePreset = getPreset(FigurePresets, figurePresetFile, keys, FigurePreset)
	print("\nYou have chosen the following preset:")
	print(figurePreset.asString(True))

	subplotPresetList = []
	print(f"You need to select {figurePreset.values['numSubFigures'].value} subplot preset(s).")
	for _ in range(figurePreset.values["numSubFigures"].value):
		keys = greeting(SubplotPresets, subplotPresetFile)
		subplotPreset = getPreset(SubplotPresets, subplotPresetFile, keys, SubplotPreset)
		print("\nYou have chosen the following preset:")
		print(subplotPreset.asString(True))
		subplotPresetList.append(subplotPreset)

	keys = greeting(AxisPresets, axisPresetFile)
	axisPresetList = []
	for index in range(figurePreset.values["numSubFigures"].value):
		temp = []
		print(f"For subplot number {index + 1} of {figurePreset.values['numSubFigures'].value}, you must choose {subplotPresetList[index].values['numSubPlots'].value} plot presets.")
		for _ in range(subplotPresetList[index].values["numSubPlots"].value):
			axisPreset = getPreset(AxisPresets, axisPresetFile, keys, AxisPreset)
			print("\nYou have chosen the following preset:")
			print(axisPreset.asString(True))
			temp.append(axisPreset)
		axisPresetList.append(temp)

	numSubplots = [len(x) for x in axisPresetList]
	
	pictureName, title, subfigureTitles, subplotTitles = getVariableValues(numSubplots)
	dataPlots = populateDatasets(axisPresetList, numSubplots)
	plotHelper(axisPresetList, subplotPresetList, figurePreset, pictureName, title, dataPlots, subfigureTitles, subplotTitles)

if __name__=="__main__":
	main()