limlam_dir = '/home/havard/Documents/covariance_calculator/limlam_mocker/'
output_dir = 'output_mcmc/'
halos_dir = 'catalogues_for_mcmc/' #'full_cita_catalogues/'#'catalogues/'
pspec_fp = 'comap_test_pspec_10muK.txt'
B_i_fp = 'bin_counts_test_10muK.txt'

mode = 'ps' #'ps'

n_patches = 1

n_walkers = 10

n_noise = 2  # Number of noise realizations per signal realization for the vid

n_realizations = 2  # Number of realizations of CO signal mapinst used to compute average power spectrum and vid
# for each mcmc-step

n_threads = 4

nsteps = 20
