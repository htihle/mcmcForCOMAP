import numpy as np
import astropy.cosmology as ac
cosmo = ac.FlatLambdaCDM(70., 0.286, Ob0=0.047)

lum_hist_bins_obs = np.logspace(5, 12, 101)
temp_hist_bins = np.logspace(1, 2, 26)

Tsys_K = 40
Nfeeds = 19
tobs_hr = 6000

limlam_dir = '/home/havard/Documents/covariance_calculator/limlam_mocker/limlam_mocker/'
output_dir = 'output_test/'
halos_dir = 'catalogues/'
pspec_fp = 'comap_test_pspec_10muK.txt'
B_i_fp = 'bin_counts_test_10muK.txt'

mode = 'ps'
n_noise = 10  # Number of noise realizations per signal realization for the vid

n_realizations = 10  # Number of realizations of CO signal map used to compute average power spectrum and vid
# for each mcmc-step

n_threads = 10

nsteps = 10
