#%%
from astropy.io import fits
import numpy 
import matplotlib.pyplot as plt
#%%

from astropy.io import fits

# Open the FITS file
filename = "2758983s.fits"
with fits.open(filename) as hdul:
    # Print the content summary
    hdul.info()
    
    # Access the binary table
    binary_table = hdul[1].data  # Typically, binary tables are in the second HDU (index 1)
    header = hdul[1].header
# View column names
print(binary_table.names)

# Access data from a specific column
wl = binary_table['Wave']
flux = binary_table['FluxAB']
error = binary_table['FluxErrAB']
sky = binary_table['FluxC']
tell_corrected = binary_table['FluxABTelluCorrected']
tell_corrected_err = binary_table['FluxErrABTelluCorrected']

plt.scatter(wl, flux, marker='.', color='black', s=1)
# plt.scatter(wl, tell_corrected, label = 'corrected')
#plt.errorbar(wl, tell_corrected, yerr=100*tell_corrected_err, fmt='')
#plt.xlim(1082.2, 1082.4)
plt.xlim(1082,1084)
plt.legend()

# Might be a titanium line 

# Control on a line that shouldn't change 
# Silicon line?

# Corrleated changes = instrumental systematics

# later on: phase coverage to look for long term trends

# email pi about plans for data 

#%%
header
# %%
error

# %%
