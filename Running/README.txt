In this file, you can find the bulk of the work. Here, the function of the contents are described:

- The "Animations" and "Figures" folders are output folders.

- The "Complexity.ipynb" file is a script that calculates the different complexity traces given a certain set of cells.
- The "Phase_Plane_Analysis.ipynb" file is a script that I wrote but didn't use; here I wanted to investigate the behaviours of different phase planes across temperature range.
- The "Plots_Producer.ipynb" file is a file with the sole purpose of producing background plots; seeing how STO behaves with certain activation variables, how spikes behave in the model etc.
- The "SimulateMultipleTemperatures.ipynb" file is one of the two most important files of the project; here, one can enter a (set of) cell(s), and a certain temperature range, and the script will run those morphologies across those temperatures. One can also change whether or not to give a current clamp, and the time or duration of it can be changed manually in the script. 
	At the end of this file, one can also find a frequency investigator. By setting the [beginning, end and step] temperatures to [32, 42 and 1], my frequency experiment can be recreated by running the cell below.
- The "UMAP_One_Cell.ipynb" file is the other important file; here, one can enter a cell and certain experiment at the top, and one can change the different parameters of UMAP (k and min_dist) or DBSCAN (epsilon or min_samples) in order to change the projection and clustering of it respectively. By changing the phas_cor variable to either True or False, a phase correction can be performed if needed.
- The "UMAP_sines.ipynb" file contains a script for the example of a UMAP clustering using sines with either linearly or randomly distributed phase shifts.
- The "Voltage_Clamps.ipynb" file contains a separate script for administering voltage clamps to a cell in steps of -70mv, -50mV and -30mV.

- All of the "~.cell.nml" files contain the morphologies of the cells acquired by Nora Vrieler.

For any questions about the scripts, feel free to reach out!