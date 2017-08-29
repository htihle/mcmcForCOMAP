import os
import sys
import shutil
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from mpi4py import MPI


import params
import experiment_params
sys.path.append(experiment_params.limlam_dir)
llm = importlib.import_module('limlam_mocker')

### begin bit from @tonyyli
import errno


def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

### end bit from @tonyyli


comm = MPI.COMM_WORLD

# print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)
my_rank, size = (comm.Get_rank(), comm.Get_size())

ensure_dir_exists(experiment_params.output_dir)
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'data'))
runid = 0
while os.path.isfile(os.path.join(
        experiment_params.output_dir, 'param', 'params_id{0:d}.py'.format(runid))):
    runid += 1
comm.Barrier()
if my_rank == 0:
    print "My run-id is ", runid
data_fp = os.path.join(
    experiment_params.output_dir, 'data', 'data_id{0:d}.npy'.format(runid))
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'cov'))
cov_fp = os.path.join(
    experiment_params.output_dir, 'cov', 'cov_mat_id{0:d}'.format(runid))
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'var_indep'))
var_indep_fp = os.path.join(
    experiment_params.output_dir, 'var_indep', 'var_indep_id{0:d}'.format(runid))
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'Nmodes'))
Nmodes_fp = os.path.join(
    experiment_params.output_dir, 'Nmodes', 'Nmodes_id{0:d}'.format(runid))
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'lum_hist'))
lum_hist_fp = os.path.join(
    experiment_params.output_dir, 'lum_hist', 'lum_hist_id{0:d}'.format(runid))
ensure_dir_exists(os.path.join(experiment_params.output_dir, 'param'))
param_fp = os.path.join(
    experiment_params.output_dir, 'param', 'params_id{0:d}.py'.format(runid))
# param_mcmc_fp = os.path.join(
#     experiment_params.output_dir, 'param', 'mcmc_params_id{0:d}.py'.format(runid))
param_experiment_fp = os.path.join(
    experiment_params.output_dir, 'param', 'experiment_params_id{0:d}.py'.format(runid))

# shutil.copy2('mcmc_params.py', param_mcmc_fp)
shutil.copy2('experiment_params.py', param_experiment_fp)
shutil.copy2('params.py', param_fp)


temp_hist_bins = experiment_params.temp_hist_bins
k_hist_bins = experiment_params.k_hist_bins
n_k = len(k_hist_bins) - 1
n_temp = len(temp_hist_bins) - 1
n_data = n_k + n_temp

full_map = llm.params_to_mapinst(params)
small_map = llm.params_to_mapinst(params)

fov_full = experiment_params.full_fov

n_maps_x = int(np.floor(fov_full / full_map.fov_x))
n_maps_y = int(np.floor(fov_full / full_map.fov_y))

full_map.fov_y = fov_full
full_map.fov_x = fov_full

full_map.pix_binedges_x = np.arange(- fov_full / 2, fov_full / 2 + full_map.pix_size_x, full_map.pix_size_x)
full_map.pix_binedges_y = np.arange(- fov_full / 2, fov_full / 2 + full_map.pix_size_y, full_map.pix_size_y)

noise_temp = experiment_params.Tsys_K * 1e6 / np.sqrt(experiment_params.tobs_hr * 3600 * experiment_params.Nfeeds /
                                                      (params.npix_x * params.npix_y) * full_map.dnu * 1e9)


CO_V = full_map.fov_x * full_map.fov_y * (np.pi / 180) ** 2 * (
    experiment_params.cosmo.comoving_transverse_distance(
        full_map.nu_rest / np.mean(full_map.nu_binedges) - 1) ** 2 * np.abs(
        experiment_params.cosmo.comoving_distance(full_map.z_i)
        - experiment_params.cosmo.comoving_distance(full_map.z_f))).value

def distribute_indices(n_indices, n_processes, my_rank):
    divide = n_indices / n_processes
    leftovers = n_indices % n_processes

    if my_rank < leftovers:
        my_n_cubes = divide + 1
        my_offset = my_rank
    else:
        my_n_cubes = divide
        my_offset = leftovers
    start_index = my_rank * divide + my_offset
    my_indices = range(start_index, start_index + my_n_cubes)
    return my_indices


def get_data(full_map, small_map, halo_fp):
    halos, cosmo = llm.load_peakpatch_catalogue(halo_fp)

    halos = llm.cull_peakpatch_catalogue(halos, params.min_mass, full_map)

    prior_ctrs = np.array((0, 1.17, 0.21, 0.3, 0.3))
    halos.Lco = llm.Mhalo_to_Lco(halos, params.model, prior_ctrs)

    lum_hist = np.histogram(np.ma.masked_invalid(halos.Lco).compressed(),
                            bins=experiment_params.lum_hist_bins_obs * 4.9e-5)[0] / np.diff(
        np.log10(experiment_params.lum_hist_bins_obs)) / CO_V

    full_map.maps = llm.Lco_to_map(halos, full_map)

    full_map.maps = full_map.maps + np.random.randn(*full_map.maps.shape) * noise_temp

    B_i = np.zeros((n_temp, n_maps_x * n_maps_y))
    ps = np.zeros((n_k, n_maps_x * n_maps_y))
    data = np.zeros((n_data, n_maps_x * n_maps_y))
    for i in range(n_maps_x):
        for j in range(n_maps_y):
            index = n_maps_x * i + j
            small_map.maps = full_map.maps[i * params.npix_x:(i + 1) * params.npix_x,
                                           j * params.npix_y:(j + 1) * params.npix_y]
            small_map.maps -= small_map.maps.mean()

            B_i[:, index] = np.histogram(small_map.maps, bins=temp_hist_bins)[0]
            _, ps[:, index], _ = llm.map_to_pspec(small_map, cosmo, kbins=k_hist_bins)
            # plt.figure()
            # plt.imshow(small_map.maps[:, :, 0], interpolation='none')

    data[:n_k] = ps
    data[n_k:n_data] = B_i
    return data, lum_hist

n_catalogues = experiment_params.n_catalogues

my_indices = distribute_indices(n_catalogues, size, my_rank)
n_catalogues_local = len(my_indices)

data = np.zeros((n_data, n_maps_x * n_maps_y, n_catalogues_local))
lum_hist = np.zeros((len(experiment_params.lum_hist_bins_obs) - 1, n_catalogues_local))
for i in range(n_catalogues_local):
    seednr = range(13579, 13901, 2)[my_indices[i]]
    halo_fp = experiment_params.limlam_dir + experiment_params.catalogue_dir + experiment_params.catalogue_name + str(seednr) + '.npz'
    data[:, :, i], lum_hist[:, i] = get_data(full_map=full_map, small_map=small_map, halo_fp=halo_fp)

gathered_data = comm.gather(data, root=0)
gathered_lum_hist = comm.gather(lum_hist, root=0)

if my_rank == 0:
    all_data = np.zeros((n_data, n_maps_x * n_maps_y, n_catalogues))
    all_lum_hist = np.zeros((len(experiment_params.lum_hist_bins_obs) - 1, n_catalogues))
    for i in range(size):
        index = distribute_indices(n_catalogues, size, i)
        all_data[:, :, index] = gathered_data[i]
        all_lum_hist[:, index] = gathered_lum_hist[i]
    all_data = all_data.reshape((n_data, n_maps_x * n_maps_y * n_catalogues))

    lum_hist_avg = all_lum_hist.mean(1)
    cov = np.cov(all_data)
    data_avg = all_data.mean(1)
    var_indep = np.zeros_like(data_avg)

    halos, cosmo = llm.load_peakpatch_catalogue(halo_fp)
    k, _, n_modes = llm.map_to_pspec(small_map, cosmo, kbins=k_hist_bins)
    var_indep[:n_k] = data_avg[:n_k] ** 2 / n_modes
    var_indep[n_k:n_data] = data_avg[n_k:n_data]

    np.save(lum_hist_fp, lum_hist_avg)
    np.save(Nmodes_fp, n_modes)

    np.save(data_fp, all_data)
    np.save(cov_fp, cov)
    np.save(var_indep_fp, var_indep)

    cov_divisor = np.sqrt(np.outer(var_indep, var_indep))
    plt.imshow(cov / cov_divisor, interpolation='none')
    plt.colorbar()

    plt.figure()
    plt.loglog(0.5 * (experiment_params.lum_hist_bins_obs[:-1] + experiment_params.lum_hist_bins_obs[1:]), lum_hist_avg)
    plt.axis([1e8, 1e12, 1e-5, 1e-1])
    plt.show()




