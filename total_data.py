#%% 
import pandas as pd 
import numpy as np 
from astropy.time import Time
from matplotlib import pyplot as plt

# Load data
data_availability = pd.read_csv('SPIRou_data.csv')

# Filter rows where "Product ID" ends with 's'
data_availability_s = data_availability[data_availability['"Product ID"'].str.endswith('s')]
data_availability_p = data_availability[data_availability['"Product ID"'].str.endswith('p')]
#%%
print(data_availability_p)
#%%
print(data_availability_s)
#%%

# Convert "Start Date" to MJD array
mjd_array = np.asarray(data_availability_s['"Start Date"'], dtype=float)  # Ensure float

# Convert MJD to UTC
utc_array = Time(mjd_array, format='mjd', scale='utc')

# Store in DataFrame as Pandas datetime
data_availability_s.loc[:, 'utc'] = utc_array.to_datetime()  # Convert to datetime64

# Sort by UTC
sorted_data = data_availability_s.sort_values(by='utc').reset_index(drop=True)

# Define histogram bins using timestamps
num_bins = 100
date_bins = np.linspace(sorted_data['utc'].min().timestamp(), 
                        sorted_data['utc'].max().timestamp(), num_bins)
date_bins = pd.to_datetime(date_bins, unit='s')  # Convert timestamps back to datetime

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(sorted_data['utc'], bins=date_bins, edgecolor='black')

# Formatting
plt.xlabel('UTC Date')
plt.ylabel('Count')
plt.title('Histogram of Observations by Date')
plt.xticks(rotation=45)
plt.grid(True)

plt.show()
#%%