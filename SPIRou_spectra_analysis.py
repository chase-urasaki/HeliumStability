#%%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from astropy.time import Time 
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.constants import c

#%%

from astropy.io import fits
# Load location, sky coordiantes, and stellar radial velocity 

CFHT = EarthLocation.of_site('Canada-France-Hawaii Telescope')
sc = SkyCoord.from_name('WASP 107')
GAMMA = 13.744*1e3 * (u.m/u.s)
#%%
# Open the FITS file
filename = "data/2955159s.fits"
with fits.open(filename) as hdul:
    # Print the content summary
    hdul.info()
    primary_header = hdul[0].header
    # Access the binary table
    binary_table = hdul[1].data  # Typically, binary tables are in the second HDU (index 1)
    header = hdul[1].header
#%%
# Get date adn time for barycentric correction

date_obs = primary_header['DATE-OBS'].strip()
time_obs = primary_header['UTIME'].strip()

obs_time = Time(f'{date_obs}T{time_obs}', scale='utc')
#%%
barycorr = sc.radial_velocity_correction(obstime=obs_time, location=CFHT)
barycorr.to(u.km/u.s)

# Compute the total doppler effect from RV and Barycentic velocity 
v_star = GAMMA - barycorr
v_star

#%%
# Function to apply barycentric coordinates to 
def apply_doppler_correction(wl, v_star): 
    wl_0 = wl/(1 + v_star/c)
    return wl_0

# Access data from a specific column
wl = binary_table['Wave']
ABflux = binary_table['FluxAB']
ABerr = binary_table['FluxErrAB']
sky = binary_table['FluxC']
ABflux_corr = binary_table['FluxABTelluCorrected']
ABerr_corr = binary_table['FluxErrABTelluCorrected']

#%%
# Only grab between certain portions of the wl range )
# Define the wavelength range
wl_min, wl_max = 1000, 1100

# Get indices where wavelengths are within the range
indices = np.where((wl >= wl_min) & (wl <= wl_max))[0]
extracted_interval_wl = wl[indices]
extracted_interval_ABflux = ABflux[indices]
extracted_interval_ABflux_corr = ABflux_corr[indices]
#%%

# He I air from NIST
HeI_1_air= 1082.909
HeI_2_air = 1083.025
HeI_3_air = 1083.034

# Functions to convert wavelengths back and forth
def air_to_vac(wl): 
    # Convert to angs
    wl_ang = wl * 10
    wl_vac = pyasl.airtovac2(wl_ang, mode="ciddor")
    wl_vac /=10
    return wl_vac 

def vac_to_air(wl):
    wl_ang = wl*10 
    wl_air = pyasl.vactoair2(wl_ang, mode="ciddor")
    wl_air /=10
    return wl_air
#%%
HeI_1_vac = air_to_vac(HeI_1_air)
HeI_2_vac = air_to_vac(HeI_2_air)
HeI_3_vac = air_to_vac(HeI_3_air)
#%%
# Plot the results: Spectrum in air and not barycenter corrected ("RAW")
plt.figure(figsize=(10,8))
plt.xlim(1080, 1085)

# Assume SPIRou is in vacuum
extracted_interval_wl_air = vac_to_air(extracted_interval_wl)

#Plot measured and line positions
plt.axvline(HeI_1_air, color='red', linestyle='dotted')
plt.axvline(HeI_2_air, color='red', linestyle='dotted')
plt.axvline(HeI_3_air, color='red', linestyle='dotted')
plt.text(HeI_1_air, 0.013, 'He I - air ', rotation=90)
plt.plot(extracted_interval_wl_air, extracted_interval_ABflux, color = 'grey', label = 'uncorrected')
plt.plot(extracted_interval_wl_air, extracted_interval_ABflux_corr, color='black', label = 'corrected')
plt.legend()
# Plot the Si I line 
Si_1_air = 1082.7091
plt.axvline(Si_1_air, color = 'green', ls = 'dotted')
plt.text(Si_1_air, 0.013, 'Si I - air', rotation = 90)
# Plot the Mg I
Mg_1_air = 1081.1084
plt.axvline(Mg_1_air, color = 'blue', ls = 'dotted')
plt.text(Mg_1_air, 0.013,  'Mg I - air', rotation=90)
plt.title('Spectrum in air and NOT barycenter corrected')
#%%
# 
# Plot Spectrum into air and shift corrected
extracted_interval_wl_air = vac_to_air(extracted_interval_wl)
plt.axvline(HeI_1_air, color='red', linestyle='dotted')
plt.axvline(HeI_2_air, color='red', linestyle='dotted')
plt.axvline(HeI_3_air, color='red', linestyle='dotted')
plt.text(HeI_1_air, 0.013, 'He I - air ', rotation=90)
# plt.axvline(HeI_measured, color='red', linestyle='--')
# plt.text(1082.97, 0.013, 'He I - Measured', rotation=90)
plt.plot(apply_doppler_correction(extracted_interval_wl_air, v_star), extracted_interval_ABflux, color = 'grey', label = 'uncorrected')
plt.plot(apply_doppler_correction(extracted_interval_wl_air, v_star), extracted_interval_ABflux_corr, color='black', label = 'corrected')
plt.legend()

# Plot the Si I line 
Si_1_air = 1082.7091
plt.axvline(Si_1_air, color = 'green', ls = 'dotted')
plt.text(Si_1_air, 0.013, 'Si I - air', rotation = 90)

# Plot the Mg I
Mg_1_air = 1081.1084
plt.axvline(Mg_1_air, color = 'blue', ls = 'dotted')
plt.text(Mg_1_air, 0.013,  'Mg I - air', rotation=90)

plt.title('Spectrum in air and doppler corrected - wavelengths match!')
#%%
# We expect to see the same results in vacuum 

# Spectrum in vacuum and barycenter corrected
#Plot measured and line positions
plt.axvline(HeI_1_vac, color='red', linestyle='dotted')
plt.axvline(HeI_2_vac, color='red', linestyle='dotted')
plt.axvline(HeI_3_vac, color='red', linestyle='dotted')
plt.text(HeI_1_vac, 0.013, 'He I - vac ', rotation=90)
# plt.axvline(HeI_measured, color='red', linestyle='--')
# plt.text(1082.97, 0.013, 'He I - Measured', rotation=90)
plt.plot(apply_doppler_correction(extracted_interval_wl, v_star), extracted_interval_ABflux, color = 'grey', label = 'uncorrected')
plt.plot(apply_doppler_correction(extracted_interval_wl, v_star), extracted_interval_ABflux_corr, color='black', label = 'corrected')
plt.legend()

# Plot the Si I line 
Si_1_air = 1082.7091
plt.axvline(air_to_vac(Si_1_air), color = 'green', ls = 'dotted')
plt.text(air_to_vac(Si_1_air), 0.013, 'Si I - vac', rotation = 90)

# Plot the Mg I
Mg_1_air = 1081.1084
plt.axvline(air_to_vac(Mg_1_air), color = 'blue', ls = 'dotted')
plt.text(air_to_vac(Mg_1_air), 0.013,  'Mg I - vac', rotation=90)

plt.title('Spectrum in vac and doppler corrected - wl matches!')
#%%

# Try fitting the Mg line first 
from specutils import Spectrum1D
# %%
# Create spectrum1d object 
spectrum = Spectrum1D(spectral_axis=apply_doppler_correction(extracted_interval_wl, v_star)*u.nm, flux = extracted_interval_ABflux_corr*u.dimensionless_unscaled)
plt.plot(spectrum.spectral_axis, spectrum.flux, label = 'original spectrum', alpha=0.7)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Flux')
from specutils.fitting import fit_generic_continuum

continuum_fit = fit_generic_continuum(spectrum)
continuum = continuum_fit(spectrum.spectral_axis)

plt.plot(spectrum.spectral_axis, continuum, label = 'continuum', ls = '--', color = 'r')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Flux')
plt.legend()
plt.xlim(1080, 1085)

#%%
# Now normalize the spectrum 
normalized_flux = spectrum.flux / continuum

plt.plot(spectrum.spectral_axis, normalized_flux, label = 'normalized flux', alpha = 0.7)
#Plot measured and line positions
plt.axvline(HeI_1_vac, color='red', linestyle='dotted')
plt.axvline(HeI_2_vac, color='red', linestyle='dotted')
plt.axvline(HeI_3_vac, color='red', linestyle='dotted')
plt.text(HeI_1_vac, 0.4, 'He I - vac ', rotation=90)
plt.axvline(air_to_vac(Si_1_air), color = 'green', ls = 'dotted')
plt.text(air_to_vac(Si_1_air), 0.4, 'Si I - vac', rotation = 90)
plt.axvline(air_to_vac(Mg_1_air), color = 'blue', ls = 'dotted')
plt.text(air_to_vac(Mg_1_air), 0.4,  'Mg I - vac', rotation=90)


plt.xlabel('Wavelength [nm]')
plt.ylabel('Normalized Flux')
plt.legend()
plt.xlim(1080, 1085)
#%%
len(spectrum.spectral_axis)

# %%
from astropy.modeling import models, fitting
wavelengths = apply_doppler_correction(extracted_interval_wl, v_star)
flux = extracted_interval_ABflux_corr
poly_init = models.Polynomial1D(degree=3)
fitter = fitting.LinearLSQFitter()
continuum_model = fitter(poly_init, wavelengths, flux)
#%%
# Evaluate the fitted continuum
continuum_flux = continuum_model(wavelengths)

#plt.scatter(wavelengths, flux, label="Original Spectrum", alpha=0.7, marker='.', s = 0.5)
plt.plot(wavelengths, flux, label="Original Spectrum", alpha=0.7)
plt.plot(wavelengths, continuum_flux, ls = '--', c='r')
plt.ylabel("Flux")
plt.legend()
plt.xlim(1080,1085)
plt.show()
# %%
#Now, normalize the spectra 
normalized_flux = flux / continuum_flux

plt.plot(wavelengths, normalized_flux, label = 'normalized flux', alpha = 0.6)
plt.ylabel('Flux')
plt.xlim(1080,1085)
plt.show()
# %%
