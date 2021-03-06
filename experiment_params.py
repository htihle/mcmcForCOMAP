import numpy as np
import astropy.cosmology as ac
cosmo = ac.FlatLambdaCDM(70., 0.286, Ob0=0.047)

lum_hist_bins_obs = np.logspace(5, 12, 101)
temp_hist_bins = np.logspace(1, 2, 26)
k_hist_bins = np.logspace(-1.5, 0, 21)  # np.linspace(1e-2, 1e0, 21)
fiducial_params = np.array((0, 1.17, 0.21, 0.3, 0.3))

include_beam = True
resol_factor = 4
fwhm = 4  # arcmin


Tsys_K = 40
Nfeeds = 19
tobs_hr = 6000

# This is the parameters for calculations of
# covariance matrices for COMAP, based on simulated
# halo catalogues.

limlam_dir = '/home/havard/Documents/covariance_calculator/limlam_mocker/'
output_dir = 'output_cov/'
catalogue_dir = 'full_cita_catalogues/'
catalogue_name = 'COMAP_z2.39-3.44_1140Mpc_seed_'

n_catalogues = 1  # 161
full_fov = 9.0  # deg. Full field of view for the simulated maps
