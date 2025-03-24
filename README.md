# Energy-Predictive-Model
Creating and displaying an AI energy predictive model based on data from historical energy consumption.

Website for Documentation: https://sites.google.com/d/1GyBltT7xm-Y9tLEhuJRRrno4hAqkmWD4/p/17yKzoxMXPiv97svOQB1UVIc9l7jHtMKb/edit

## Main Files to run: 
MapWindow.py -> This file contains the Python to load and display the Map Data GUI.
PredictionModelv4.ipynb -> This file runs the predictions and puts them in the nc_energy.db database file. 

## Exporting to Tableau:
Use the nc_energy_combined.csv file to import the combined tables in the SQLite database (predictions and historical data) into Tableau. This works on any version of Tableau including Tableau Public. 

## Dataset files:
historic-census-2.csv -> This file contains the population data for every County in North Carolina from 1990 to 2020 with 10 yr intervals.
nc_energy.db -> This is the SQLite database with 4 tables, all normalized for ease of access and querying. This is what the prediction model reads for its predictions.
NCCountyBoundaries.geojson -> this file contains the borders for displaying the counties on the map for the MapWindow.py file.
energy-and-utilities-linc.csv -> origninal dataset with energy consumption data that is formatted and placed in the nc_energy.db dataset.

## Other Files:
temp_plot.html -> Displays the plots using html and a WebEngine
temp_map.html -> Displays the map using html and a WebEngine 
debug_output.txt -> this is the error log file used for debugging (not needed to run the project).
