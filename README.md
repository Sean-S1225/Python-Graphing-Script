# Python-Graphing-Script
A python script created to take data from TSV files and graph them using Seaborn.

Specifically, each row in each file to graph is expected to have the data for that index in the second column.

The first version of this script was written to generate of graphs of data from the Thayer Lab, but I found it quite tedious to graph different types of datasets. For example, RMSD graphs and RMSF graphs have a number of differences, and it was difficult to remember exaclty which lines of code to comment and which to uncomment.

This script was designed to avoid this problem, by allowing the user of the script to have custom Presets, groups of settings, saved to a file, so that each time they want to plot a graph, all they need to do is specify the name of which Preset they want to use.

Additionally, I coded some interface code, allowing the user of the script to modify and create their Presets without needing to directly modify the JSON file (although that certainly is an option).

The Preset file that I use is uploaded for reference.
