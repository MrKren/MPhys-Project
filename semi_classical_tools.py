import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps


def lone_electron_R(w, t):
    """
    A function for generating the spin correlation tensor for a singular electron using the exact quantum mechanical
    calculation
    :param w: array representing the [x,y,z] components of the external magnetic field
    :param t: array representing the the timescale in mT^-1 to generate the result for eg np.linspace(0, 30, 170)
    :return:
    """
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1  # Levi-Civita Tensor
    dij = np.identity(3)  # Kronecker delta
    omega = np.linalg.norm(w)
    w /= omega

    s = 0.5
    R = np.zeros([3,3,len(t)])
    for alpha in range(3):
        for beta in range(3):
            R[alpha][beta] += w[alpha]*w[beta] + (dij[alpha, beta] - w[alpha]*w[beta])*np.cos(omega*t)
            for gamma in range(3):
                R[alpha][beta] += eijk[alpha, beta, gamma]*w[gamma]*np.sin(omega*t)

    R *= s*(s+1)*(2*s+1)/3

    return R


def singlet_prob(R_1, R_2):
    product = np.zeros(R_1.shape[-1])

    for alpha in range(3):
        for beta in range(3):
            product += R_1[alpha][beta]*R_2[alpha][beta]

    return 0.25 + product


def quantum_yield(k, p, t):

    y = p*np.exp(-k*t)

    return k*simps(y, t)


def electronic_density_operator(R_1, R_2):
    p = np.zeros([4, 4, len(R_1[0, 0])], dtype=complex)

    # Convert R to complex type
    R_1 = R_1.astype(complex)
    R_2 = R_2.astype(complex)

    # Spin operators
    s_x = np.zeros([2, 2], dtype=complex)
    s_x[0, 1] = s_x[1, 0] = 0.5
    s_y = np.zeros([2, 2], dtype=complex)
    s_y[1, 0] = 0.5j
    s_y[0, 1] = -0.5j
    s_z = np.identity(2, dtype=complex)
    s_z[1, 1] = -1
    s_z *= 0.5
    spin_op = [s_x, s_y, s_z]

    # Create p
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                s_A_s_b = np.kron(spin_op[alpha], spin_op[beta])
                s_A_s_b = np.repeat(s_A_s_b, len(R_1[0, 0]), axis=1).reshape((4,4,len(R_1[0, 0])))
                p -= 4 * s_A_s_b * R_1[gamma][alpha]*R_2[gamma][beta]

    p += 0.25*np.repeat(np.identity(4, dtype=complex), len(R_1[0, 0]), axis=1).reshape((4,4,len(R_1[0, 0])))

    return p


def coherence_measure(p):
    p = abs(p)

    return np.sum(p, axis=(0, 1)) - np.trace(p)


def collect_singlet_probs(file_path, t, title, plot=True):
    probabilities = []

    plt.figure(figsize=(9, 6))
    with open(file_path) as file:
        lines = file.readlines()
        lines = lines[5:]
        index = 0
        for u in np.linspace(-1, 1, 10):
            for theta in np.linspace(0, 2 * np.pi, 10):
                r_flavin = np.load(lines[index][:-1] + '.npy')
                mag_field = np.array(
                    [np.sqrt(1 - u ** 2) * np.cos(theta), np.sqrt(1 - u ** 2) * np.sin(theta), u]) * 0.05
                r_0 = lone_electron_R(mag_field, t)

                s_p = singlet_prob(r_flavin, r_0)
                probabilities.append([s_p, u, theta, t])
                if plot:
                    plt.plot(t, s_p)
                index += 1

    plt.xlabel('Time (mT^-1)')
    plt.ylabel('Singlet Probability')
    plt.title('Singlet Probability of ' + title)
    if plot:
        plt.show()

    return probabilities
