import csv
from genericpath import isfile
from logging.config import valid_ident
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pandas as pd
import os.path
import json
from typing import Callable, TypeVar, Generic
import copy
import pprint

A = TypeVar("A")
B = TypeVar("B")

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

class Value:
	"""A way of abstracting each value the user might want to input to their preset.
	Avoids code duplication, and makes it easier to add/remove values in the future.
	"""
	def __init__(self, name: str, newPrompt: str, modifyPrompt: str, modifyCommand: str, help: str, display: str, verify: Callable[[str], bool], convert: Callable[[str], any], value: any) -> None:
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

Values = {
    "name": Value("name", "What should the name of your preset be? ", "name [New Name]", "name ",
        "The name of your preset. Only used to help identify and distinguish from other presets.", "Name: ", lambda name: type(name) == str, lambda name: name, ""),
    "numRows": Value("numRows", "How many rows are in the largest file? ", "numRows [Number of Rows]", "numRows ",
        "The number of rows in the largest file you want to graph. This just helps speed up read-in of the files.", "Number of Rows: ", lambda numRows: numRows.isdigit(), lambda numRows: int(numRows), 0),
    "numPlots": Value("numPlots", "How many datasets are you looking to plot? ", "numPlots [Number of Datasets]", "numPlots ",
        "The number of sets of data you want to graph; the number of files you want to graph.", "Number of Plots: ", lambda numPlots: numPlots.isdigit(), lambda numPlots: int(numPlots), 0),
    "xAxisTitle": Value("xAxisTitle", "What should the title of the X-Axis be? ", "xAxisTitle [Title]", "xAxisTitle ",
        "The title of the X-Axis", "Title of X-Axis: ", lambda xAxisTitle: type(xAxisTitle) == str, lambda xAxisTitle: xAxisTitle, ""),
    "yAxisTitle": Value("yAxisTitle", "What should the title of the Y-Axis be? ", "yAxisTitle [Title]", "yAxisTitle ",
        "The title of the Y-Axis", "Title of Y-Axis: ", lambda yAxisTitle: type(yAxisTitle) == str, lambda yAxisTitle: yAxisTitle, ""),
    "movAvg": Value("movAvg", "Would you like a moving average? (Y/N) ", "movAvg [Y | N]", "movAvg ",
        "A boolean determining if you want a moving average, useful for \"smoothing out\" noisy data.", "Moving Average? ", lambda movAvg: movAvg in ["Y", "N"], lambda movAvg: True if movAvg == "Y" else False, False),
    "movAvgFr": Value("movAvgFr", "How many frames would you like your moving average to be? ", "movAvgFr [Number of Frames]", "movAvgFr ",
        "The number of frames by which to calculate the moving average. If movAvg = False, then this is ignored.", "Moving Average Frames: ", lambda movAvgFr: movAvgFr.isdigit(), lambda movAvgFr: int(movAvgFr), 0),
    "palette": Value("palette", "Please enter the palette you would like to use, hex values separated only by a space. ", "palette [Hex Values separated by only a space]", "palette ", 
        "Hex values determining the color of each dataset. If you are using a moving average, you should input 2*numPlots, in the order: [plot1] [mov avg plot1], etc", "Palette: ", lambda palette: reduce(lambda s: s[0] == "#" and len(s) == 7, True, lambda x, y: x and y, palette.split(" ")), lambda palette: palette.split(" "), []),
    "xLimit": Value("xLimit", "Please enter the upper and lower X-Bounds, separated by only a space. ", "xLimit [[Lower Bound] [Upper Bound]]", "xLimit ",
        "Determines the X-Axis view limits. If left > right, then the X-Axis values will decrease from left to right.", "X Bounds: ", lambda xLimit: (len(xLimit.split(" ")) == 2 and reduce(lambda s: s.isdigit(), True, lambda x, y: x and y, xLimit.split(" "))), lambda xLimit: [int(x) for x in xLimit.split(" ")], []),
    "xTicks": Value("xTicks", "Please enter the lower bound, upper bound, and frequency of ticks, each separated by one space. ", "xTicks [[Lower Bound] [Upper Bound] [Frequency]]", "xTicks ",
        "Determines where the ticks on the X-Axis begin and end, and how frequent they are.", "X Ticks: ", lambda xTicks: (len(xTicks.split(" ")) == 3 and reduce(lambda s: s.isdigit(), True, lambda x, y: x and y, xTicks.split(" "))), lambda xTicks: [int(x) for x in xTicks.split(" ")], []),
    "xScale": Value("xScale", "What should the scale of the X-Axis be? ", "xScale [Scale]", "xScale ",
        "One unit on the X-Axis should be equal to how many units in the data?", "X Scale: ", lambda xScale: xScale.isdigit(), lambda xScale: int(xScale), 1)
}

class Dataset:
	"""An object that stores the data and relevant information from each file:
	name (string): The name assigned to this set of data,
	length (int): The number of rows in the file, used to initialize the list to avoid repeatedly appending,
	data (int list list): The actual data stored in the file, Data[n][0]: row number, Data[n][1]: value associated with its row.
	movAvg (int list): If the user wants a moving average, stores the value of that average, starting at index "amount"-1, leaving those indices None.
	amount (int): The number of frames for the moving average.
	"""
	def __init__(self, length):
		self.name, self.path = None, None
		self.length = length
		self.data = [None] * length
		self.movAvg = []

	def Prompt(self):
		"""
		Gets the name and path to a dataset
		"""
		self.name = input("What should the name of this data be? ")
		self.path = input("What\'s the path to " + self.name + "? ")
		while not os.path.isfile(self.path):
			print("That is not a valid file.")
			self.path = input("What\'s the path to " + self.name + "? ")

	def Populate(self):
		"""
		Reads in the provided path and stores all the data in self.data
		"""
		print(self.path)
		with open(self.path) as file:
			rd = csv.reader(file, delimiter="\t")
			for index, value in enumerate(rd):
				self.data[index] = float(value[1])

	def CalcMovingAvg(self, amount):
		"""
		Calculates a moving average, given amount frames
		"""
		self.movAvg = [None] * self.length
		self.amount = amount
		for x in range(amount-1, self.length):
			avg = 0
			for y in range(x - (amount-1), x+1):
				avg += self.data[y]
			avg /= amount
			self.movAvg[x] = avg

	def AsDataFrame(self):
		"""
		Converts the Dataset to a DataFrame format
		"""
		if self.movAvg:
			dataFrame = pd.DataFrame({self.name : self.data, f"{self.name} {self.amount} Frame\nMov Avg" : self.movAvg})
		else:
			dataFrame = pd.DataFrame({self.name : self.data})
		return dataFrame

class Preset:
    """
    An established group of settings by which to graph data.
    """
    def __init__(self, name: str, numRows: int, numPlots: int, xAxisTitle: str, yAxisTitle: str, movAvg: bool, movAvgFr: int, palette: list[str], xLimit: list[int], xTicks: list[str], xScale: int):
        """Initializes all the values of the Preset

        Args:
            name (str): Name of the Preset
            numRows (int): The number of rows in the largest file
            numPlots (int): The number of datasets to plot
            xAxisTitle (str): Title of X-Axis
            yAxisTitle (str): Title of Y-Axis
            movAvg (bool): Does the user want a moving average?
            movAvgFr (int): How many frames for the moving average
            palette (list[str]): List of hex values for palette
            xLimit (list[int]): List containing lower and upper bounds for X-Axis
            xTicks (list[str]): List containing lower, upper bounds for ticks, and frequency
            xScale (int): Scale to multiply the X-Axis by
        """
        self.values = copy.deepcopy(Values)
        self.values["name"].value = name
        self.values["numRows"].value = numRows
        self.values["numPlots"].value = numPlots
        self.values["xAxisTitle"].value = xAxisTitle
        self.values["yAxisTitle"].value = yAxisTitle
        self.values["movAvg"].value = movAvg
        self.values["movAvgFr"].value = movAvgFr
        self.values["palette"].value = palette
        self.values["xLimit"].value = xLimit
        self.values["xTicks"].value = xTicks
        self.values["xScale"].value = xScale

    def __str__(self) -> str:
        """Returns the Preset as a string of its values

        Returns:
            str: String of Preset values, each on a new line with title
        """
        toReturn = ""
        for x in list(self.values.values()):
            toReturn += f"{x.display}{x.value}\n"
        return toReturn

    def __repr__(self):
        return self.__str__()

    def getValues(self) -> dict[str, any]:
        """Returns a dictionary containing the name of each value, and the value associated with it

        Returns:
            dict[str, any]: A dictionary containing the name and value associated with each value in the Preset
        """
        return {n.name: n.value for n in list(self.values.values())}

    def copy(self):
        """Returns a copy of itself

        Returns:
            Preset: Copy of itself
        """
        return Preset(self.values["name"].value, self.values["numRows"].value, self.values["numPlots"].value, self.values["xAxisTitle"].value, self.values["yAxisTitle"].value, self.values["movAvg"].value, self.values["movAvgFr"].value, self.values["palette"].value, self.values["xLimit"].value, self.values["xTicks"].value, self.values["xScale"].value)

def new(Presets, presetFile):
    """
    Handles when the user wants to create a new Preset.

    Returns the newly created Preset.
    """
    preset = Preset("", 0, 0, "", "", False, 0, [], [], [], 1)

    print("Note:\n\
If you choose to save your file and you have not chosen a unique name, it will override the original.\n")

    #Get user input for each of the values
    for x in list(preset.values.values()):
        userIn = input(x.newPrompt)
        while not x.verify(userIn):
            print("Invalid input.")
            userIn = input(x.newPrompt)
        x.value = x.convert(userIn)

    save = input("Would you like to save this preset? (Y/N) ")
    while save not in ["Y", "N"]:
        print("Invalid input.")
        save = input("Would you like to save this preset? (Y/N) ")

    if save == "Y":
        #converts this preset to a dictionary
        dict = {preset.values["name"].value: preset.getValues()}
        #Converts all the other presets to dictionaries and combines them all together
        for x in list(Presets.values()):
            dict |= {x.values["name"].value: x.getValues()}

        with open(presetFile, "w", encoding="utf-8") as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)

    return preset

def modify(Presets, presetFile) -> Preset:
    """
    Handles when the user wants to modify an existing Preset.

    Returns the modified Preset. Does not modify original Preset.
    """
    modify = input("Which preset would you like to modify? ")
    while modify not in Presets:
        print("Invalid input.")
        modify = input()

    #Need to create a copy otherwise all the values will just be overwritten
    oldPreset = Presets[modify].copy()

    print("You can use the following commands to modify the preset:")
    for x in list(oldPreset.values.values()):
        print(x.modifyPrompt)

    print("Type \"done\" when you are done.\n\
If you choose to save your file and you have not chosen a unique name, it will override the original.\n\
Otherwise it will save as a copy of the original.")

    #Get user input to change values of preset
    command = input()
    while(command != "done"):
        for x in list(oldPreset.values.keys()):
            if command[:len(oldPreset.values[x].modifyCommand)] == oldPreset.values[x].modifyCommand:
                if oldPreset.values[x].verify(command[len(oldPreset.values[x].modifyCommand):]):
                    oldPreset.values[x].value = oldPreset.values[x].convert(command[len(oldPreset.values[x].modifyCommand):])
                else:
                    print("Invalid input.")

        command = input()

    save = input("Would you like to save this preset? (Y/N) ")
    while save not in ["Y", "N"]:
        print("Invalid input.")
        save = input("Would you like to save this preset? (Y/N) ")

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

def greeting(Presets, presetFile) -> Preset:
	"""
	Greets the user, determines which Preset they would like to use, if
	they would like to create a new Preset, or modify an existing Preset.

	Returns the Preset they have chosen to use.
	"""
	keys = list(Presets.keys())

	if keys != []:
		print("Here are your loaded presets:\n")
		for preset in keys:
			print(Presets[preset])
			print("")
		print("Please enter the name of the preset you would like to use.\n\
Enter \"new\" if you would like to create a new preset from scratch,\n\
or enter \"modify\" if you would like to edit an existing preset.\n\
You will be prompted to enter the title of the picture, the title of the graph, and the subtitle later.")
	else:
		print(f"You have no presets on file. If you believe you should, check to find a {presetFile} file in this directory.")
		print("Enter \"new\" to create a new preset from scratch.")

	keys.append("new")
	keys.append("modify")

	preset = input()
	while preset not in keys or (preset == "modify" and Presets == {}):
		print("Invalid input.")
		preset = input()

	if preset == "new":
		preset = new(Presets, presetFile)
	elif preset == "modify":
		preset = modify(Presets, presetFile)
	else:
		preset = Presets[preset]

	return preset

def getVariableValues() -> tuple[str, str, str]:
    """Gets input from user that are likely to vary each time the script is run

    Returns:
        tuple[str, str, str]: The name of the picture, the title of the graph, the subtitle of the graph
    """
    pictureName = input("What should the picture of the plot be named? ")
    title = input("What should the title of the graph be? ")
    suptitle = input("What should the subtitle of the graph be? ")
    return (pictureName, title, suptitle)

def populateDatasets(preset: Preset) -> pd.DataFrame:
    """Populates each dataset by getting input from the user

    Args:
        preset (Preset): List of all the presets loaded

    Returns:
        pd.DataFrame: Dataframe containing all the data for the graph
    """
    plots = []

    dataPlot = pd.DataFrame({})
    for x in range(preset.values["numPlots"].value):
        x = Dataset(preset.values["numRows"].value)
        x.Prompt()
        plots.append(x)

    for x in plots:
        x.Populate()
        # print("Preset value", preset.values["movAvg"].value, type(preset.values["movAvg"].value))
        if preset.values["movAvg"].value:
            x.CalcMovingAvg(preset.values["movAvgFr"].value)
        dataPlot = pd.concat([dataPlot, x.AsDataFrame()], axis = 1)

    dataPlot.reset_index(inplace=True)
    dataPlot["index"] = dataPlot["index"].apply(lambda x: x/preset.values["xScale"].value)

    dataPlot = pd.melt(dataPlot, ["index"])
    print(dataPlot)
    return dataPlot

def plot(preset, pictureName, title, suptitle, dataPlot):
	plt.figure(figsize = (6, 4))
	ax = sns.lineplot(data = dataPlot, x = "index", y = "value", dashes = False, hue = "variable", palette = preset.values["palette"].value)
	ax.set(xlabel = preset.values["xAxisTitle"].value, ylabel = preset.values["yAxisTitle"].value)
	plt.suptitle(title, ha="center", x = 0.515)
	plt.title(suptitle, fontsize="small", alpha=0.8, ha="center", x = 0.5)
	plt.legend(borderaxespad = 0)
	sns.move_legend(ax, bbox_to_anchor = (.5, -.15 ), ncol = 3, loc = "upper center", edgecolor = "white")
	plt.grid(visible = True, color = "#D9D9D9")
	ax.set_xlim(preset.values["xLimit"].value[0], preset.values["xLimit"].value[1])
	ax.set_xticks(range(preset.values["xTicks"].value[0], preset.values["xTicks"].value[1], preset.values["xTicks"].value[2]))
	plt.ticklabel_format(style='plain')
	sns.set(font = "Helvetica")
	ax.figure.savefig(pictureName, bbox_inches = "tight", dpi = 500)

def loadPresets(Presets: dict[str, Preset], presetFile: str) -> dict[str, Preset]:
    """Reads in any presets from presetFile and appends them to a dictionary.

    If the file does not exist, function throws a warning to the user, and returns an empty dictionary.

    Args:
        Presets (dict[str, Preset]): An empty dictionary
        presetFile (str): A string to where the Preset JSON file can be located

    Returns:
        dict[str, Preset]: The dictionary containing the name, and the associated Preset. Empty if Preset file does not exist.
    """
    data = {}
    if os.path.isfile(presetFile):
        with open(presetFile) as file:
            data = json.load(file)
        for x in list(data.keys()):
            Presets = Presets | {x: Preset(data[x]["name"], data[x]["numRows"], data[x]["numPlots"], data[x]["xAxisTitle"], data[x]["yAxisTitle"], data[x]["movAvg"], data[x]["movAvgFr"], data[x]["palette"], data[x]["xLimit"], data[x]["xTicks"], data[x]["xScale"])}

    return Presets

def main():
	presetFile = "presets.json"
	Presets = {}

	Presets = loadPresets(Presets, presetFile)
	preset = greeting(Presets, presetFile)
	print("\nYou have chosen the following preset:")
	print(preset)
	pictureName, title, suptitle = getVariableValues()
	dataPlot = populateDatasets(preset)
	plot(preset, pictureName, title, suptitle, dataPlot)

if __name__=="__main__":
    main()