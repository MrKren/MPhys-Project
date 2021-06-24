import numpy as np
import matplotlib.pyplot as plt
import semi_classical_tools as sct
import os
import plotter as p


def field(direction):
    if direction == "X":
        return [0.05, 0, 0]
    if direction == "Y":
        return [0, 0.05, 0]
    if direction == "Z":
        return [0, 0, 0.05]


if __name__ == '__main__':

    # Constants
    gamma_e = 0.176085963023 * 1e3  # For microseconds
    T_100us_mT = np.linspace(0, 600*30, 600*170)
    T_1ms_mT = np.linspace(0, 6000*30, 6000*170)
    T_10us_mT = np.linspace(0, 60 * 30, 60 * 170)
    T_1us_mT = np.linspace(0, 6 * 30, 6 * 170)
    T_0_15us_mT = np.linspace(0, 1 * 30, 1 * 170)

    # Importing and creating SCT's
    file_path = 'SCT/'
    files = os.listdir(file_path)
    R = {name[:-4]: np.load(file_path + name) for name in files}
    R_10 = {name[:-4]: p.load_R((file_path + name), 10) for name in files}
    R_100 = {name[:-4]: p.load_R((file_path + name), 100) for name in files}
    R_600 = {name[:-4]: p.load_R((file_path + name), 600) for name in files}
    Z_0_15us = [sct.lone_electron_R(i, T_0_15us_mT) for i in [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]]
    Z_1us = [sct.lone_electron_R(i, T_1us_mT) for i in [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]]
    Z_10us = [sct.lone_electron_R(i, T_10us_mT) for i in [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]]
    Z_100us = [sct.lone_electron_R(i, T_100us_mT) for i in [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]]
    Z_1ms = [sct.lone_electron_R(i, T_1ms_mT) for i in [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]]

    # Singlet Probability
    p.singlet_prob_plot(max(T_10us_mT)/gamma_e,
                        (R_10['flavin_anion_No_Bath_Z'], Z_10us[2]),
                        (R_10['flavin_anion_No_Bath_Y'], Z_10us[1]),
                        names=["Z", "Y"], title=r"")

    p.singlet_prob_plot(max(T_10us_mT) / gamma_e,
                        (R_10['flavin_anion_No_Bath_Z'], Z_10us[2]),
                        (R_10['flavin_anion_Drosophila_Z'], Z_10us[2]),
                        (R_10['flavin_anion_Pigeon_Resting_Z'], Z_10us[2]),
                        (R_10['flavin_anion_Pigeon_Radical_Z'], Z_10us[2]),
                        names=["No Bath", "Drosophila", "Pigeon Resting", "Pigeon Radical"], title=r"")

    p.singlet_prob_plot(max(T_10us_mT)/gamma_e,
                        (R_10['flavin_anion_Drosophila_X'], Z_10us[0]),
                        (R_10['flavin_anion_Drosophila_Y'], Z_10us[1]),
                        (R_10['flavin_anion_Drosophila_Z'], Z_10us[2]),
                        names=["X", "Y", "Z"], title=r'')

    fig, axs = plt.subplots(3, sharex='col', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(12,12))
    axs[0].plot(T_10us_mT/gamma_e, sct.singlet_prob(R_10['flavin_anion_Drosophila_X'], Z_10us[0]))
    axs[0].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Drosophila_Y'], Z_10us[1]))
    axs[0].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Drosophila_Z'], Z_10us[2]))
    axs[0].legend(["X", "Y", "Z"], fontsize='xx-large')
    axs[0].set_title('(A)', y=0.8, fontsize='xx-large')

    axs[1].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Resting_X'], Z_10us[0]))
    axs[1].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Resting_Y'], Z_10us[1]))
    axs[1].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Resting_Z'], Z_10us[2]))
    axs[1].set_title('(B)', y=0.8, fontsize='xx-large')

    axs[2].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Radical_X'], Z_10us[0]))
    axs[2].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Radical_Y'], Z_10us[1]))
    axs[2].plot(T_10us_mT / gamma_e, sct.singlet_prob(R_10['flavin_anion_Pigeon_Radical_Z'], Z_10us[2]))
    axs[2].set_title('(C)', y=0.8, fontsize='xx-large')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(r"Singlet Probability", fontsize='xx-large')
    plt.xlabel(r"Time ($\mu s$)", fontsize='xx-large')
    plt.show()


    # Quantum Yield
    no_bath = ['flavin_anion_No_Bath_X', 'flavin_anion_No_Bath_Y', 'flavin_anion_No_Bath_Z']
    dros = ['flavin_anion_Drosophila_X', 'flavin_anion_Drosophila_Y', 'flavin_anion_Drosophila_Z']
    res = ['flavin_anion_Pigeon_Resting_X', 'flavin_anion_Pigeon_Resting_Y', 'flavin_anion_Pigeon_Resting_Z']
    rad = ['flavin_anion_Pigeon_Radical_X', 'flavin_anion_Pigeon_Radical_Y', 'flavin_anion_Pigeon_Radical_Z']
    bath_names = [no_bath, dros, res, rad]

    ks = np.logspace(np.log10(1 / (500 / 15)), 2, 100)

    yields = {}
    for i in bath_names:
        for j in i:
            sp = sct.singlet_prob(R[j], sct.lone_electron_R(field(j[-1]), T_100us_mT))
            yields[j] = [sct.quantum_yield(k, sp, T_100us_mT/gamma_e) for k in ks]

    names = [i[-1] for i in bath_names]
    p.yield_plot(np.log10(ks), [yields[i] for i in names],
                 names=[j.replace('_', ' ')[:-2] for j in [i.replace('flavin_anion_', '') for i in names]])

    yield_diff = {}
    for i in bath_names:
        data = [yields[j] for j in i]
        yield_min = np.min(np.vstack(data), axis=0)
        yield_max = np.max(np.vstack(data), axis=0)
        yield_diff[i[0][:-2]] = yield_max - yield_min

    p.yield_plot(np.log10(ks), [yield_diff[i] for i in yield_diff],
                 names=[j.replace('_', ' ') for j in [i.replace('flavin_anion_', '') for i in yield_diff]], ylabel=r'Anisotropy in $\Phi_S$')

    [print(ks[list(j).index(max(j))]) for j in [yield_diff[i] for i in yield_diff]]

    # Coherence Plot
    p.coherence_plot(max(T_0_15us_mT) / gamma_e,
                     (R_600['flavin_anion_No_Bath_Z'], Z_0_15us[2]),
                     (R_600['flavin_anion_Drosophila_Z'], Z_0_15us[2]),
                     (R_600['flavin_anion_Pigeon_Resting_Z'], Z_0_15us[2]),
                     (R_600['flavin_anion_Pigeon_Radical_Z'], Z_0_15us[2]),
                     names=["No Bath", "Drosophila", "Pigeon Resting", "Pigeon Radical"], title=r"")

    p.coherence_plot(max(T_1us_mT)/gamma_e,
                     (R_100['flavin_anion_No_Bath_Z'], Z_1us[2]),
                     (R_100['flavin_anion_Drosophila_Z'], Z_1us[2]),
                     (R_100['flavin_anion_Pigeon_Resting_Z'], Z_1us[2]),
                     (R_100['flavin_anion_Pigeon_Radical_Z'], Z_1us[2]),
                     names=["No Bath", "Drosophila", "Pigeon Resting", "Pigeon Radical"], title=r"")

    p.coherence_plot_expo(max(T_1us_mT) / gamma_e, 1,
                     (R_10['flavin_anion_No_Bath_Z'], Z_10us[2]),
                     (R_10['flavin_anion_Drosophila_Z'], Z_10us[2]),
                     (R_10['flavin_anion_Pigeon_Resting_Z'], Z_10us[2]),
                     (R_10['flavin_anion_Pigeon_Radical_Z'], Z_10us[2]),
                     names=["No Bath", "Drosophila", "Pigeon Resting", "Pigeon Radical"], title=r"")

    # Spin bath structure

    p.singlet_prob_plot(max(T_10us_mT) / gamma_e,
                        (R_10['flavin_anion_Pigeon_Radical_Z'], Z_10us[2]),
                        (R_10['pigeon_radical_500_spins'], Z_10us[2]),
                        (R_10['pigeon_radical_200_spins'], Z_10us[2]),
                        (R_10['pigeon_radical_100_spins'], Z_10us[2]),
                        names=["All spins", "500 spins", "200 spins", "100 spins"], title=r"")

    p.singlet_prob_plot(max(T_10us_mT) / gamma_e,
                        (R_10['flavin_anion_Pigeon_Resting_Z'], Z_10us[2]),
                        (R_10['pigeon_resting_500_spins'], Z_10us[2]),
                        (R_10['pigeon_resting_200_spins'], Z_10us[2]),
                        (R_10['pigeon_resting_100_spins'], Z_10us[2]),
                        names=["All spins", "500 spins", "200 spins", "100 spins"], title=r"")

    p.coherence_plot(max(T_1us_mT) / gamma_e,
                        (R_100['flavin_anion_Pigeon_Radical_Z'], Z_1us[2]),
                        (R_100['pigeon_radical_500_spins'], Z_1us[2]),
                        (R_100['pigeon_radical_200_spins'], Z_1us[2]),
                        (R_100['pigeon_radical_100_spins'], Z_1us[2]),
                        names=["All spins", "500 spins", "200 spins", "100 spins"], title=r"")

    p.coherence_plot(max(T_1us_mT) / gamma_e,
                        (R_100['flavin_anion_Pigeon_Resting_Z'], Z_1us[2]),
                        (R_100['pigeon_resting_500_spins'], Z_1us[2]),
                        (R_100['pigeon_resting_200_spins'], Z_1us[2]),
                        (R_100['pigeon_resting_100_spins'], Z_1us[2]),
                        names=["All spins", "500 spins", "200 spins", "100 spins"], title=r"")

    datarest = np.load(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Michael\bardataresting.npz')
    datarad = np.load(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Michael\bardataradical.npz')

    fig = plt.figure(figsize=(12,8))
    plt.bar(datarad['trad'], datarad['heightrad'], width=0.102, alpha=0.75)
    plt.bar(datarest['trest'], datarest['heightrest'], width=0.102, alpha=0.6)
    plt.xlabel('Log$_{10}$(hyperfine strength)')
    plt.ylabel('Number density')
    plt.legend(['Radical', 'Resting'])
    plt.show()


