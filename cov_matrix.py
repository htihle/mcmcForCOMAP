import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from mpi4py import MPI

import limlam_mocker as llm
import params
import cov_params
import mcmc_params


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

ensure_dir_exists(cov_params.output_dir)
ensure_dir_exists(os.path.join(cov_params.output_dir, 'data'))
runid = 0
while os.path.isfile(os.path.join(
        cov_params.output_dir, 'data', 'data_id{0:d}.npy'.format(runid))):
    runid += 1
data_fp = os.path.join(
    cov_params.output_dir, 'data', 'data_id{0:d}.npy'.format(runid))
ensure_dir_exists(os.path.join(cov_params.output_dir, 'cov'))
cov_fp = os.path.join(
    cov_params.output_dir, 'cov', 'cov_mat_id{0:d}'.format(runid))
ensure_dir_exists(os.path.join(cov_params.output_dir, 'param'))
param_fp = os.path.join(
    cov_params.output_dir, 'param', 'params_id{0:d}.py'.format(runid))
param_mcmc_fp = os.path.join(
    cov_params.output_dir, 'param', 'mcmc_params_id{0:d}.py'.format(runid))
param_cov_fp = os.path.join(
    cov_params.output_dir, 'param', 'cov_params_id{0:d}.py'.format(runid))
shutil.copy2('mcmc_params.py', param_mcmc_fp)
shutil.copy2('cov_params.py', param_cov_fp)
shutil.copy2('params.py', param_fp)

temp_hist_bins = mcmc_params.temp_hist_bins #np.logspace(1, 2, 26)

map = llm.params_to_mapinst(params)


fov_full = cov_params.full_fov

n_maps_x = int(np.floor(fov_full/map.fov_x))
n_maps_y = int(np.floor(fov_full/map.fov_y))

map.fov_y = fov_full
map.fov_x = fov_full

map.pix_binedges_x = np.arange(- fov_full / 2, fov_full / 2 + map.pix_size_x, map.pix_size_x)
map.pix_binedges_y = np.arange(- fov_full / 2, fov_full / 2 + map.pix_size_y, map.pix_size_y)

noise_temp = mcmc_params.Tsys_K * 1e6 / np.sqrt(mcmc_params.tobs_hr * 3600 * mcmc_params.Nfeeds /
                                                (params.npix_x * params.npix_y) * map.dnu * 1e9)

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


def get_temp_histograms(map, halo_fp):
    halos, cosmo = llm.load_peakpatch_catalogue(halo_fp)

    halos = llm.cull_peakpatch_catalogue(halos, params.min_mass, map)

    prior_ctrs = np.array((0, 1.17, 0.21, 0.3, 0.3))
    halos.Lco = llm.Mhalo_to_Lco(halos, params.model, prior_ctrs)

    map.maps = llm.Lco_to_map(halos, map)

    map.maps = map.maps + np.random.randn(*map.maps.shape)
    map.maps -= map.maps.mean()

    B_i = np.zeros((len(temp_hist_bins) - 1, n_maps_x * n_maps_y))

    for i in range(n_maps_x):
        for j in range(n_maps_y):
            index = n_maps_x * i + j
            B_i[:, index] = np.histogram(map.maps[i * params.npix_x:(i + 1) * params.npix_x,
                                                  j * params.npix_y:(j + 1) * params.npix_y],
                                         bins=temp_hist_bins)[0]
            # plt.figure()
            # plt.imshow(map.maps[i * n_pix_small:(i + 1) * n_pix_small, j * n_pix_small:(j + 1) * n_pix_small, 0], interpolation='none')
    return B_i

n_catalogues = cov_params.n_catalogues

my_indices = distribute_indices(n_catalogues, size, my_rank)
n_catalogues_local = len(my_indices)

B_i = np.zeros((len(temp_hist_bins) - 1, n_maps_x * n_maps_y, n_catalogues_local))
for i in range(n_catalogues_local):
    seednr = range(13579, 13901, 2)[my_indices[i]]
    halo_fp = cov_params.catalogue_dir + cov_params.catalogue_name + str(seednr) + '.npz'
    B_i[:, :, i] = get_temp_histograms(map=map, halo_fp=halo_fp)

gathered_data = comm.gather(B_i, root=0)

if my_rank == 0:
    all_data = np.zeros((len(temp_hist_bins) - 1, n_maps_x * n_maps_y, n_catalogues))
    for i in range(size):
        all_data[:, :, distribute_indices(n_catalogues, size, i)] = gathered_data[i]
    all_data = all_data.reshape((len(temp_hist_bins) - 1, n_maps_x * n_maps_y * n_catalogues))

    cov = np.cov(all_data)
    B_i_avg = all_data.mean(1)
    cov_divisor = np.sqrt(np.outer(B_i_avg, B_i_avg))

    np.save(data_fp, all_data)
    np.save(cov_fp, cov)
    plt.imshow(cov / cov_divisor, interpolation='none')
    plt.show()




