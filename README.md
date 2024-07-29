# README for Risk Analysis Project

## Overview

This project analyzes and visualizes the relationship between inspired oxygen levels (L/min) and O2 saturation levels (chronic respiratory) with modeled risk values. The analysis uses a combination of interpolation, mesh grid creation, and polynomial surface fitting to derive and visualize a risk surface.

## Data Source

The data used in this project is sourced from an Excel file named `DATA.xlsx`. This file contains columns for:
- `Inspired oxygen (L/min)`
- `O2 sats (chronic resp)`
- `Modelled risk`

However the data in these columns can be any numeric data relating to risk.

## Steps in Analysis

1. **Data Loading and Preparation:**
   - Load the data from the Excel file using `pandas`.
   - Separate the data into two sets: This can be any numerical data for this example duringt the code I used one for inspired oxygen and one for O2 saturation. 
   - Filter the data within specified ranges for inspired oxygen (0.5 to 15 L/min) and O2 saturation (50% to 100%). These filtres are dependant on the data that you wish to use into the model.

2. **Normalization:**
   - Normalize the `Modelled risk` values to be between 0 and 1 for both data sets.

3. **Grouping:**
   - Group the data into bins for inspired oxygen (0.5 increments) and O2 saturation (5% increments).
   - Calculate the average risk value within each group.

4. **Interpolation:**
   - Create interpolation functions for both inspired oxygen and O2 saturation to model risk values.

5. **Mesh Grid Creation:**
   - Create a mesh grid for the range of inspired oxygen and O2 saturation values.
   - Calculate the combined risk using the interpolated values and a specific risk combination formula.

6. **Polynomial Surface Fitting:**
   - Flatten the mesh grid data for polynomial fitting.
   - Fit a polynomial surface to the combined risk data using `scipy.optimize.curve_fit`.

7. **Visualization:**
   - Generate a 3D surface plot of the fitted risk values.
   - Customize the plot with labels, reversed axes, and a color bar for better visualization.

## Requirements

The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `mpl_toolkits.mplot3d`

Ensure these libraries are installed in your environment. You can install them using `pip`:
```sh
pip install pandas numpy matplotlib scipy
