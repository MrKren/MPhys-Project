"""semi_classical_script.py: Uses semi-classical approximation methods to calculate electron spin correlation tensors"""

__author__ = "Kieran Dominy"
__credits__ = ["Matt Cooke", "Kieran Dominy", "Michael Harris", "Daniel Kattnig", "Charlotte Remnant"]
__version__ = "1.0.1"
__maintainer__ = "Kieran Dominy"
__email__ = "kad212@exeter.ac.uk"
__status__ = "Production"

import numpy as np
from scipy.integrate import odeint
import multiprocessing as mp
import timeit
import platform
from datetime import datetime


def sphere_sampling_func(r_values, is_radical=True):
    """
    Uniformly samples random points on a sphere
    :param is_radical: A boolean to determine if a electron spin should be added to the sphere sampling
    :param r_values: 1D array of spin numbers for the nuclei in molecule: 1 for nitrogen, 1/2 for hydrogen etc.
    :return: 2D array of vectors of electron and nuclear spins ie [[s_x,s_y,s_z],[i1_x,i1_y,i1_z],...]
    """

    if is_radical:
        r_values = np.insert(r_values, 0, 0.5)  # for initial electron spin
    vectors = []
    for r in r_values:
        length = np.sqrt(r * (r + 1))
        u = (np.random.random_sample() - 0.5) * 2
        theta = np.random.random_sample() * 2 * np.pi
        vec = np.array([np.sqrt(1 - u ** 2) * np.cos(theta), np.sqrt(1 - u ** 2) * np.sin(theta), u])
        vec = length * vec
        vectors.append(vec)

    return np.array(vectors)


def ode_solver(y0, t, omega_vec, a):
    """
    Solves the necessary semi-classical differential equations using the given initial conditions
    :param y0: 1D array of vector components for the initial conditions
    :param t: 1D array of the timespace to solve over
    :param omega_vec: 3 long array representing the external magnetic field vector
    :param a: Array of 3x3 matrices containing the hyperfine coupling constants between each nucleus and the
    electron in units of mT
    :return: An array representing S(t)
    """
    y0 = y0.reshape(len(y0) * 3)

    def ddt(y, t, omega_vec, a):
        ans = np.zeros([len(y) // 3, 3])

        s = y[:3]
        i = y[3:]
        i = i.reshape(len(i) // 3, 3, 1)
        omega = omega_vec + np.sum((a@i).reshape((len(y)//3)-1, 3), axis=0)

        ans[0] += np.cross(omega, s)

        i = i.reshape((len(y)//3)-1, 3)
        ans[1:] += np.cross(a@s, i)

        return ans.reshape(len(y))

    sol = odeint(ddt, y0, t, args=(omega_vec, a))
    return sol[:, :3]


def data_packer(spins, t, omega_vec, a, resolution, bath=None):
    """
    Zips ODE data together for the purposes of multiprocessing
    :param bath: file containing magnetic field contribution of a static spin bath
    :param spins: 1D array of spin numbers for the nuclei in molecule: 1 for nitrogen, 1/2 for hydrogen etc.
    :param t: 1D array of the timespace to solve over
    :param omega_vec: 3 long array representing the external magnetic field vector
    :param a: Array of 3x3 matrices containing the hyperfine coupling constants between each nucleus and the
    electron in units of mT
    :param resolution: The number of iterations to use in the integration
    :return: A zipped iterator that contains the necessary variables for multiprocessing
    """

    t = np.tile(t, (resolution,) + tuple(np.repeat(1, t.ndim)))
    samples = np.zeros([resolution, (len(spins) + 1), 3])  # An empty array for filling of random initial conditions
    for index, sample in enumerate(samples):
        sample = sphere_sampling_func(spins)
        samples[index] += sample
    omega_vec = np.tile(omega_vec, (resolution,) + tuple(np.repeat(1, omega_vec.ndim)))
    if bath is not None:
        omega_vec += bath[:resolution]
    a = np.tile(a, (resolution,) + tuple(np.repeat(1, a.ndim)))

    return zip(samples, t, omega_vec, a)


def monte_carlo_int_mp(y0, t, omega_vec, a):
    """
    A function to calculate the electron spin correlation tensor for a specific trajectory
    :param y0: 1D array of vector components for the initial conditions
    :param t: 1D array of the timespace to solve over
    :param omega_vec: 3 long array representing the external magnetic field vector
    :param a: Array of 3x3 matrices containing the hyperfine coupling constants between each nucleus and the
    electron in units of mT
    :return:
    """
    spin_corr = np.zeros([3, 3, len(t)])
    s = ode_solver(y0, t, omega_vec, a)
    for alpha in range(len(spin_corr)):
        for beta in range(len(spin_corr[0])):
            spin_corr[alpha][beta] += 2 * s[0][alpha] * s[:, beta]

    return spin_corr


def integrate(spins, t, omega_vec, a, resolution: int, bath=None):
    """
    A function that manages the multiprocessing of the Monte Carlo integration
    :param bath:
    :param spins: 1D array of spin numbers for the nuclei in molecule: 1 for nitrogen, 1/2 for hydrogen etc.
    :param t: 1D array of the timespace to solve over in units of mT^-1
    :param omega_vec: 3 long array representing the external magnetic field vector
    :param a: Array of 3x3 matrices containing the hyperfine coupling constants between each nucleus and the
    electron in units of mT
    :param resolution: The number of iterations to use in the integration
    :return:
    """
    st = timeit.default_timer()
    print("Performing Monte Carlo integration")
    print(f"Processing {resolution} trajectories with: {mp.cpu_count() - 1} cores")
    with mp.Pool(mp.cpu_count() - 1) as pool:
        R_vals = pool.starmap(monte_carlo_int_mp, data_packer(spins, t, omega_vec, a, resolution, bath=bath))
    print(f"Total time taken: {round(timeit.default_timer() - st, 2)} s")

    R = np.zeros([3, 3, len(t)])
    for val in R_vals:
        R += val

    return R / resolution


def generate_bath_field(spins, a):
    """
    Generates a magnetic field contribution from a static spin bath
    :param spins: nuclear spin values of atoms in bath
    :param a: corresponding hyperfine coupling tensors for nuclear spins
    :return:
    """
    vec = sphere_sampling_func(spins, is_radical=False)
    vec = vec.reshape([len(spins), 3, 1])
    return np.sum((a @ vec).reshape((len(spins), 3)), axis=0)


def bath_fields_mp(resolution, spins, a):
    """
    A wrapper function for multiprocessing the generate_bath_field function
    :param resolution: Number of bath fields ot generate
    :param spins: nuclear spin values of atoms in bath
    :param a: corresponding hyperfine coupling tensors for nuclear spins
    :return:
    """
    st = timeit.default_timer()
    vals = zip(np.tile(spins, (resolution,) + tuple(np.repeat(1, spins.ndim))),
               np.tile(a, (resolution,) + tuple(np.repeat(1, a.ndim))))

    print("Generating random bath fields")
    print(f"Processing {resolution} samples with: {mp.cpu_count() - 1} cores")
    with mp.Pool(mp.cpu_count() - 1) as pool:
        bath_fields = pool.starmap(generate_bath_field, vals)
    print(f"Total time taken: {round(timeit.default_timer() - st, 2)} s")

    return np.array(bath_fields)


if __name__ == '__main__':
    # Initial parameters
    radical_name = "flavin_anion_radical_bath.npz"
    radical = np.load(radical_name)
    t_scale = 600  # ~100 us
    T = np.linspace(0, t_scale*30, t_scale*170)
    samples = int(1e4)

    # Generate Bath
    bath = np.load('bath_values.npz')
    bath_spins = bath['spins']
    bath_a = bath['a']
    bath_fields = bath_fields_mp(samples, bath_spins, bath_a)

    # Monte Carlo integration
    R = integrate(radical['spins'], T, radical['omega_vec'], radical['a'], samples, bath=bath_fields)

    save_name = f"{radical_name[:-4]}_res_{samples}_t_{t_scale*30}_{platform.node()}_{datetime.now()}"
    np.save(save_name, R)
