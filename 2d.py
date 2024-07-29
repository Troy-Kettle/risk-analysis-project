import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Load the data from the updated Excel file
file_path = 'Test7.xlsx'  # Update with the correct file path if needed
data = pd.read_excel(file_path)

# Separate the data into two sets: Inspired oxygen and O2 sats (chronic resp)
inspired_oxygen_data = data[['Inspired oxygen (L/min)', 'Modelled risk']].dropna()
o2_sats_data = data[['O2 sats (chronic resp)', 'Modelled risk']].dropna()

# Limit the data to specified ranges
inspired_oxygen_data = inspired_oxygen_data[(inspired_oxygen_data['Inspired oxygen (L/min)'] >= 0.5) & (inspired_oxygen_data['Inspired oxygen (L/min)'] <= 15)]
o2_sats_data = o2_sats_data[(o2_sats_data['O2 sats (chronic resp)'] >= 50) & (o2_sats_data['O2 sats (chronic resp)'] <= 100)]

# Normalize risk values to be between 0 and 1
inspired_oxygen_data['Modelled risk'] = (inspired_oxygen_data['Modelled risk'] - inspired_oxygen_data['Modelled risk'].min()) / (inspired_oxygen_data['Modelled risk'].max() - inspired_oxygen_data['Modelled risk'].min())
o2_sats_data['Modelled risk'] = (o2_sats_data['Modelled risk'] - o2_sats_data['Modelled risk'].min()) / (o2_sats_data['Modelled risk'].max() - o2_sats_data['Modelled risk'].min())

# Group the data into specified sections
inspired_oxygen_bins = np.arange(0.5, 16, 1)
o2_sats_bins = np.arange(50, 101, 5)

inspired_oxygen_data['Inspired Oxygen Group'] = pd.cut(inspired_oxygen_data['Inspired oxygen (L/min)'], bins=inspired_oxygen_bins, labels=inspired_oxygen_bins[:-1] + 0.5)
o2_sats_data['O2 Sats Group'] = pd.cut(o2_sats_data['O2 sats (chronic resp)'], bins=o2_sats_bins, labels=o2_sats_bins[:-1] + 2.5)

# Average the risk values within each group
inspired_oxygen_grouped = inspired_oxygen_data.groupby('Inspired Oxygen Group')['Modelled risk'].mean().reset_index()
o2_sats_grouped = o2_sats_data.groupby('O2 Sats Group')['Modelled risk'].mean().reset_index()

# Remove any rows with NaN values that may have been created during the grouping process
inspired_oxygen_grouped = inspired_oxygen_grouped.dropna()
o2_sats_grouped = o2_sats_grouped.dropna()

# Create interpolation function for Inspired oxygen and O2 sats
inspired_oxygen_interp = interp1d(inspired_oxygen_grouped['Inspired Oxygen Group'].astype(float), inspired_oxygen_grouped['Modelled risk'], kind='linear', fill_value="extrapolate")
o2_sats_interp = interp1d(o2_sats_grouped['O2 Sats Group'].astype(float), o2_sats_grouped['Modelled risk'], kind='linear', fill_value="extrapolate")

# Create a mesh grid for Inspired oxygen and O2 sats
inspired_oxygen_range = np.linspace(0.5, 15, 100)
o2_sats_range = np.linspace(50, 100, 100)
inspired_oxygen_grid, o2_sats_grid = np.meshgrid(inspired_oxygen_range, o2_sats_range)

# Calculate the combined risk (using a different method)
inspired_oxygen_risk = inspired_oxygen_interp(inspired_oxygen_grid)
o2_sats_risk = o2_sats_interp(o2_sats_grid)
combined_risk = 1 - (1 - inspired_oxygen_risk) * (1 - o2_sats_risk)

# Ensure the risk values are between 0 and 1
combined_risk = np.clip(combined_risk, 0, 1)

# Flatten the data for polynomial fitting
X = inspired_oxygen_grid.flatten()
Y = o2_sats_grid.flatten()
Z = combined_risk.flatten()

# Check for any inf or NaN values in the arrays
if np.any(np.isnan(X)) or np.any(np.isnan(Y)) or np.any(np.isnan(Z)):
    raise ValueError("Input arrays contain NaN values")
if np.any(np.isinf(X)) or np.any(np.isinf(Y)) or np.any(np.isinf(Z)):
    raise ValueError("Input arrays contain inf values")

# Define the polynomial function
def polynomial_surface(XY, a, b, c, d, e, f):
    x, y = XY
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

# Fit the polynomial surface using curve_fit
popt, _ = curve_fit(polynomial_surface, (X, Y), Z)

# Generate the fitted surface
Z_fit = polynomial_surface((inspired_oxygen_grid, o2_sats_grid), *popt)

# Ensure the fitted values are between 0 and 1
Z_fit = np.clip(Z_fit, 0, 1)

# Plotting the fitted surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(inspired_oxygen_grid, o2_sats_grid, Z_fit, cmap='viridis', alpha=0.7)

ax.set_xlabel('Inspired oxygen (L/min)')
ax.set_ylabel('O2 sats (chronic resp)')
ax.set_zlabel('Risk')

# Reverse the x-axis (Inspired oxygen) and align with z-axis minimum
ax.set_xlim(15, 0.5)
ax.set_zlim(0, 1)

# Adjust the view angle so the lower ends of x and z axes are next to each other
ax.view_init(elev=30, azim=135)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
