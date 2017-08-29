import os
import sys
import shutil
import random
import emcee
import warnings
import importlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline

import mcmc_params

if len(sys.argv) < 2:
    import params
    import experiment_params
    cov_mode = 'diag'
else:
    cov_id = sys.argv[1]
    cov_mode = 'full'
    sys.path.append('output_cov/param/')
    experiment_params = importlib.import_module('experiment_params_id' + str(cov_id))
    params = importlib.import_module('params_id' + str(cov_id))
    cov_mat_full = np.load('output_cov/cov/cov_mat_id' + str(cov_id) + '.npy')
    full_data = np.load('output_cov/data/data_id' + str(cov_id) + '.npy')
    if mcmc_params.mode == 'ps':
        cov_mat = cov_mat_full[:len(experiment_params.k_hist_bins) - 1, :len(experiment_params.k_hist_bins) - 1]
        var_indep_0 = np.load('output_cov/var_indep/var_indep_id' +
                              str(cov_id) + '.npy')[:len(experiment_params.k_hist_bins) - 1]
        data = full_data[:len(experiment_params.k_hist_bins) - 1, 0]
    elif mcmc_params.mode == 'vid':
        cov_mat = cov_mat_full[len(experiment_params.k_hist_bins) - 1:, len(experiment_params.k_hist_bins) - 1:]
        var_indep_0 = np.load('output_cov/var_indep/var_indep_id' +
                              str(cov_id) + '.npy')[len(experiment_params.k_hist_bins) - 1:]
        data = full_data[len(experiment_params.k_hist_bins) - 1:, 0]
    elif mcmc_params.mode == 'vid + ps':
        cov_mat = cov_mat_full
        var_indep_0 = np.load('output_cov/var_indep/var_indep_id' +
                              str(cov_id) + '.npy')
        data = full_data[:, 0]
    else:
        print "Unknown mode. Only 'vid', 'ps', 'vid + ps' are allowed."
        print "You gave:", mcmc_params.mode
        print "Please insert valid mode in mcmc_params.py"
        sys.exit()
    # plt.imshow(cov_mat / np.sqrt(np.outer(var_indep_0, var_indep_0)), interpolation='none')
    # plt.show()
    # data_avg_0 = full_data.mean(1)
    # np.savetxt('bin_counts_test_10muK.txt', (0.5 * (mcmc_params.temp_hist_bins[:-1] + mcmc_params.temp_hist_bins[1:]), full_data[:, 33]))


sys.path.append(mcmc_params.limlam_dir)
llm = importlib.import_module('limlam_mocker')

lnnormal = lambda x, mu, sigma: -(x - mu) ** 2 / (2 * sigma ** 2)

mode = mcmc_params.mode
n_noise = mcmc_params.n_noise
n_realizations = mcmc_params.n_realizations
n_threads = mcmc_params.n_threads
mapinst = llm.params_to_mapinst(params)
CO_V = mapinst.fov_x * mapinst.fov_y * (np.pi / 180) ** 2 * (
    experiment_params.cosmo.comoving_transverse_distance(
        mapinst.nu_rest / np.mean(mapinst.nu_binedges) - 1) ** 2 * np.abs(
        experiment_params.cosmo.comoving_distance(mapinst.z_i)
        - experiment_params.cosmo.comoving_distance(mapinst.z_f))).value

lum_hist_bins_obs = experiment_params.lum_hist_bins_obs
temp_hist_bins = experiment_params.temp_hist_bins
k_hist_bins = experiment_params.k_hist_bins
noise_temp = experiment_params.Tsys_K * 1e6 / np.sqrt(experiment_params.tobs_hr * 3600 * experiment_params.Nfeeds /
                                                      (params.npix_x * params.npix_y) * mapinst.dnu * 1e9)
print "Noise temperature per pixel, ", noise_temp, "muK"
lum_hist_bins_int = lum_hist_bins_obs * 4.9e-5

all_halos = []
halo_dir_list = os.listdir(mcmc_params.limlam_dir + mcmc_params.halos_dir)
for i in range(len(halo_dir_list)):
    halos_fp = os.path.join(mcmc_params.limlam_dir + mcmc_params.halos_dir, halo_dir_list[i])
    halos, cosmo = llm.load_peakpatch_catalogue(halos_fp)
    all_halos.append(llm.cull_peakpatch_catalogue(halos, params.min_mass, mapinst))
print "All halos loaded!"


def noise_ps(Tsys, Nfeeds, tobs, Oobs, fwhm, Ompix, dnu, Dnu,
              z, cosmo, nu_rest, Nmodes):
    # Tsys in K; tobs in sec
    # Oobs in sr; fwhm, dpix in rad
    # dnu (channel BW), Dnu (total BW), nu_rest all in GHz
    #dk = np.mean(np.diff(k))  # 1/Mpc
    ctd = cosmo.comoving_transverse_distance(z).value
    dx_fwhm = ctd * fwhm / 2.355  # Mpc
    dz = 299792.458 / cosmo.H(z).value * dnu / nu_rest * (1 + z) ** 2  # Mpc
    Dz = 299792.458 / cosmo.H(z).value * Dnu / nu_rest * (1 + z) ** 2  # Mpc
    Pn = 1e3 * Tsys ** 2 / (Nfeeds * dnu * tobs * Ompix / Oobs) * ctd ** 2 * Ompix * dz
    # if Nmodes is None:
    #     Nmodes = k ** 2 * dk * Oobs * ctd ** 2 * Dz / (4 * np.pi ** 2)
    # mu = np.linspace(0, 1, 201)[:, None]
    # W_integrand = np.exp(k ** 2 * (dx_fwhm ** 2 - dz ** 2) * mu ** 2)
    # W = np.exp(-k ** 2 * dx_fwhm ** 2) * np.trapz(W_integrand, mu, axis=0)
    return Pn / np.sqrt(Nmodes), Pn  #, Nmodes , W  # (Pk + Pn) / np.sqrt(Nmodes) / W, Pn, Nmodes, W

if cov_mode == 'diag':
    x, B_i_data = np.loadtxt(mcmc_params.B_i_fp)
if (mode == 'ps') or (mode == 'vid + ps'):
    if cov_mode == 'diag':
        k_tofit, Pk_tofit, Nmodes = np.loadtxt(mcmc_params.pspec_fp)
    elif cov_mode == 'full':
        Nmodes = np.load('output_cov/Nmodes/Nmodes_id' + str(cov_id) + '.npy')
    sigma_noise, Pnoise = noise_ps(experiment_params.Tsys_K,
                                   experiment_params.Nfeeds, experiment_params.tobs_hr * 3600,
                                   mapinst.fov_x * mapinst.fov_y * (np.pi / 180) ** 2,
                                   np.pi / (15 * 180), mapinst.Ompix, mapinst.dnu,
                                   np.ptp(mapinst.nu_binedges),
                                   mapinst.nu_rest / np.mean(mapinst.nu_binedges) - 1,
                                   experiment_params.cosmo, mapinst.nu_rest,
                                   Nmodes=Nmodes)


def generate_mock_map(pos):
    global mapinst
    # halos_fp = os.path.join(mcmc_params.limlam_dir + mcmc_params.halos_dir,
    #                         random.choice(os.listdir(mcmc_params.limlam_dir + mcmc_params.halos_dir)))
    # halos, cosmo = llm.load_peakpatch_catalogue(halos_fp)
    # halos = llm.cull_peakpatch_catalogue(halos, params.min_mass, full_map2)
    halos = all_halos[np.random.randint(0, len(all_halos))]
    halos.Lco = llm.Mhalo_to_Lco(halos, params.model, pos)
    if np.all(np.isfinite(halos.Lco)):
        lum_hist = np.histogram(halos.Lco, bins=lum_hist_bins_int)[0] / np.diff(
            np.log10(lum_hist_bins_obs)) / CO_V
    else:
        lum_hist = np.histogram(np.ma.masked_invalid(halos.Lco).compressed(),
                                bins=lum_hist_bins_int)[0] / np.diff(
            np.log10(lum_hist_bins_obs)) / CO_V
    mapinst.maps = llm.Lco_to_map(halos, mapinst)

    # Add noise
    map_with_noise = mapinst.maps[None, :] + np.random.randn(n_noise, *mapinst.maps.shape) * noise_temp
    map_with_noise -= map_with_noise.mean(axis=(1, 2, 3))[:, None, None, None]
    B_i = np.histogram(np.ma.masked_invalid(map_with_noise).compressed(),
                       bins=temp_hist_bins)[0] / n_noise
    k, Pk, Nmodes = llm.map_to_pspec(mapinst, cosmo, kbins=k_hist_bins)
    return Pk, Pk / np.sqrt(Nmodes), lum_hist, B_i


def lnprior(pos):
    log_delta_mf, alpha, beta, sigma_sfr, sigma_lco = pos
    if sigma_sfr < 0 or sigma_lco < 0:
        return -np.inf
    return (lnnormal(log_delta_mf, 0., 0.3) +
            lnnormal(alpha, 1.17, 0.37) + lnnormal(beta, 0.21, 3.74) +
            lnnormal(sigma_sfr, 0.3, 0.1) + lnnormal(sigma_lco, 0.3, 0.1))


def lnlike(pos):
    if pos[-1] < 0 or pos[-2] < 0:
        return -np.inf, np.nan * np.ones_like(k_hist_bins[1:]), np.nan * np.ones_like(
            lum_hist_bins_obs), np.nan * np.ones_like(
            temp_hist_bins)
    if pos[1] == 0:
        return -np.inf, np.nan * np.ones_like(k_hist_bins[1:]), np.nan * np.ones_like(
            lum_hist_bins_obs)
    Pk_mod = np.zeros((n_realizations, len(k_hist_bins) - 1))
    sigma_sample = np.zeros((n_realizations, len(k_hist_bins) - 1))
    lum_hist = np.zeros((n_realizations, len(lum_hist_bins_obs) - 1))
    B_i = np.zeros((n_realizations, len(temp_hist_bins) - 1))
    for i in range(n_realizations):
        Pk_mod[i], sigma_sample[i], lum_hist[i], B_i[i] = generate_mock_map(pos)
    Pk_mod = Pk_mod.mean(0)
    sigma_sample = sigma_sample.mean(0)
    lum_hist = lum_hist.mean(0)
    B_i = B_i.mean(0)
    B_i[np.where(B_i == 0)] = 0.01
    if cov_mode == 'diag':
        if mode == 'ps':
            loglike = -0.5 * np.sum(
                (Pk_tofit - Pnoise - Pk_mod) ** 2 / (sigma_noise ** 2 + (n_realizations + 1.0)/n_realizations * sigma_sample ** 2)
                + np.log(sigma_noise ** 2 + (n_realizations + 1.0)/n_realizations * sigma_sample ** 2))
        elif mode == 'vid':
            loglike = -np.sum((B_i - B_i_data) ** 2 / (2 * B_i) + np.log(B_i))
            # warnings.filterwarnings('error')
            # try:
            #     data[np.where(data == 0)] = 0.01
            #     loglike = -np.sum((data - B_i_data) ** 2 / (2 * data) + np.log(data))
            # except RuntimeWarning:
            #     print data
            #     print "RuntimeWarning caught, returning - np.inf"
            #     print "pos = ", pos
            #     loglike = - np.inf
            # warnings.filterwarnings('default')

        else:
            print "Unknown, mode"
            loglike = - np.infty
    else:
        if mode == 'ps':
            mean = Pk_mod + Pnoise
            var_indep = mean ** 2 / Nmodes
        elif mode == 'vid':
            mean = B_i
            var_indep = mean
        elif mode == 'vid + ps':
            n_k = len(k_hist_bins) - 1
            n_data = len(data)
            mean = np.zeros_like(data)
            mean[:n_k] = Pk_mod + Pnoise
            mean[n_k:n_data] = B_i
            var_indep = np.zeros_like(var_indep_0)
            var_indep[:n_k] = mean[:n_k] ** 2 / Nmodes
            var_indep[n_k:n_data] = mean[n_k:n_data]

        local_cov_mat = cov_mat * np.sqrt(np.outer(var_indep, var_indep) / np.outer(var_indep_0, var_indep_0))
        inv_cov_mat = np.linalg.inv(local_cov_mat)
        local_cov_det = np.linalg.det(local_cov_mat)

        if local_cov_det <= 0:
            return -np.infty, Pk_mod, lum_hist, B_i
        loglike = - 0.5 * (np.matmul((data - mean), np.matmul(inv_cov_mat, (data - mean))) + np.log(local_cov_det))

    return loglike, Pk_mod, lum_hist, B_i


def lnprob(pos):
    ll = lnlike(pos)

    result = lnprior(pos) + ll[0]
    if not np.isfinite(result):
        return -np.inf, ll[1:]
    return result, ll[1:]

### begin bit from @tonyyli
import errno


def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


### end bit from @tonyyli


if __name__ == '__main__':
    ensure_dir_exists(mcmc_params.output_dir)
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'chain'))
    runid = 0
    while os.path.isfile(os.path.join(
            mcmc_params.output_dir, 'param', 'mcmc_params_run{0:d}.py'.format(runid))):
        runid += 1

    print "My runid is: ", runid
    chain_fp = os.path.join(
        mcmc_params.output_dir, 'chain', 'run{0:d}.dat'.format(runid))
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'pspec'))
    pspec_fp = os.path.join(
        mcmc_params.output_dir, 'pspec', 'run{0:d}.dat'.format(runid))
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'lumif'))
    lumif_fp = os.path.join(
        mcmc_params.output_dir, 'lumif', 'run{0:d}.dat'.format(runid))
    lum_avg_fp = os.path.join(
        mcmc_params.output_dir, 'lumif', 'avg_run{0:d}.npy'.format(runid))
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'data'))
    B_i_fp = os.path.join(
        mcmc_params.output_dir, 'data', 'run{0:d}.dat'.format(runid))
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'acorr'))
    with open(lumif_fp, 'a') as flumif:
        flumif.write('# lum bins: np.logspace(5,12,101)\n # mode =' + mode + '\n')
    acorr_fp = os.path.join(
        mcmc_params.output_dir, 'acorr', 'run{0:d}.dat'.format(runid))
    ensure_dir_exists(os.path.join(mcmc_params.output_dir, 'param'))
    param_fp = os.path.join(
        mcmc_params.output_dir, 'param', 'params_run{0:d}.py'.format(runid))
    mcmc_param_fp = os.path.join(
        mcmc_params.output_dir, 'param', 'mcmc_params_run{0:d}.py'.format(runid))
    experiment_param_fp = os.path.join(
        mcmc_params.output_dir, 'param', 'experiment_params_run{0:d}.py'.format(runid))
    shutil.copy2('mcmc_params.py', mcmc_param_fp)
    shutil.copy2('output_cov/param/' + 'experiment_params_id' + str(cov_id) + '.py', experiment_param_fp)
    shutil.copy2('output_cov/param/' + 'params_id' + str(cov_id) + '.py', param_fp)
    shutil.copy2('output_cov/lum_hist/' + 'lum_hist_id' + str(cov_id) + '.npy', lum_avg_fp)
    n_walkers = mcmc_params.n_walkers
    # if mode == 'ps':
    #     sampler = emcee.EnsembleSampler(n_walkers, 5, lnprob, threads=n_threads)
    # elif mode == 'vid':
    #     sampler = emcee.EnsembleSampler(n_walkers, 5, lnprob, threads=n_threads)
    sampler = emcee.EnsembleSampler(n_walkers, 5, lnprob, threads=n_threads)
    prior_ctrs = np.array((0, 1.17, 0.21, 0.3, 0.3))
    pos = [prior_ctrs + 1e-3 * np.random.randn(5) for i in range(n_walkers)]
    i = 0
    while i < mcmc_params.nsteps:
        print('undergoing iteration {0}'.format(i))
        for result in sampler.sample(pos, iterations=1, storechain=True):
            position, _, _, blobs = result
            try:
                tacorr = sampler.acor
            except:
                tacorr = np.nan
            print('recording iteration {0}'.format(i))
            with open(chain_fp, 'a') as (fchain), \
                    open(pspec_fp, 'a') as (fpspec), \
                    open(lumif_fp, 'a') as (flumif), \
                    open(B_i_fp, 'a') as (fB_i), \
                    open(acorr_fp, 'a') as facorr:
                j = 0
                while j < n_walkers:  # nwalkers
                    fchain.write('{0:4d} {1:s}\n'.format(
                        j, ' '.join([str(p) for p in position[j]])))
                    fpspec.write('{0:4d} {1:s}\n'.format(
                        j, ' '.join([str(b) for b in blobs[j][0]])))
                    flumif.write('{0:4d} {1:s}\n'.format(
                        j, ' '.join([str(b) for b in blobs[j][1]])))
                    fB_i.write('{0:4d} {1:s}\n'.format(
                        j, ' '.join([str(b) for b in blobs[j][2]])))
                    j += 1
                facorr.write('{0:s}\n'.format(str(tacorr)))
        pos = position
        i += 1
