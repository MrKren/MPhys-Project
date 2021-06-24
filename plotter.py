import matplotlib.pyplot as plt
import numpy as np
import semi_classical_tools as sct
import os


def spin_corr_plot(T, *args, names=(), title="Electron spin correlation tensor"):

    fig, axs = plt.subplots(3, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(9,6))
    labels = ('x', 'y', 'z')

    for R in args:
        t = np.linspace(0, T, len(R[0, 0]))

        for i in range(3):
            for j in range(3):
                axs[i, j].plot(t, R[i][j])
                if j == 0:
                    axs[i, j].set_ylabel(r'$R_{\alpha ' + str(labels[i]) + r'}$')
                if i == 2:
                    axs[i, j].set_xlabel(r'$R_{ ' + str(labels[j]) + r'\beta}$')

    axs[0, 2].legend(names)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r"Time ($\mu s$)", labelpad=20)
    plt.title(title)
    plt.show()


def singlet_prob_plot(T, *args, names=(), title="Singlet Probability"):
    fig = plt.figure(figsize=(12, 8))

    for R in args:
        R_1, R_2 = R
        t = np.linspace(0, T, len(R_1[0, 0]))
        sp = sct.singlet_prob(R_1, R_2)
        plt.plot(t, sp)

    plt.xlabel(r"Time ($\mu s$)", fontsize='xx-large')
    plt.ylabel(r"Singlet Probability", fontsize='xx-large')
    plt.legend(names, fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.title(title)
    plt.show()


def yield_plot(k, args, names=(), ylabel=r"$\Phi_S$"):
    fig = plt.figure(figsize=(12, 8))

    for qy in args:
        plt.plot(k, qy)

    plt.xlabel(r"log$_{10}$(k)", fontsize='xx-large')
    plt.ylabel(ylabel, fontsize='xx-large')
    plt.legend(names, fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.show()


def coherence_plot(T, *args, names=(), title="Coherence measure"):
    fig = plt.figure(figsize=(12, 8))

    for R in args:
        R_1, R_2 = R
        p = sct.electronic_density_operator(R_1, R_2)
        c = sct.coherence_measure(p)
        t = np.linspace(0, T, len(c))
        plt.plot(t, c)

    plt.xlabel(r'Time ($\mu s$)', fontsize='xx-large')
    plt.ylabel(r'Coherence measure', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    # plt.legend(names, bbox_to_anchor=(0.5, 1.10), loc='upper center', ncol=len(names),
    #            shadow=True, fancybox=True, fontsize='xx-large')
    plt.legend(names, fontsize='xx-large')
    plt.title(title)
    plt.show()


def coherence_plot_expo(T, k, *args, names=(), title=""):
    fig = plt.figure(figsize=(12, 8))

    for R in args:
        R_1, R_2 = R
        p = sct.electronic_density_operator(R_1, R_2)
        c = sct.coherence_measure(p)
        t = np.linspace(0, T, len(c))
        c_expo = c*np.exp(-k*t)
        plt.plot(t, c_expo)

    plt.xlabel(r'Time ($\mu s$)', fontsize='xx-large')
    plt.ylabel(r'Coherence measure', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.legend(names, fontsize='xx-large')
    plt.title(title)
    plt.show()


def load_R(filename, n):
    R = np.load(filename)
    x = int(len(R[0][0])/n)

    return R[:,:,:x]


if __name__ == '__main__':
    gamma_e = 0.176085963023*1e3

    N = 100
    T = np.linspace(0, int(600 * 30 / N), int(600 * 170 / N))
    R_no_bath = load_R(r"C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\Final Results\SCT\flavin_anion_No_Bath_Z.npy", N)
    R_resting = load_R(r"C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\Final Results\SCT\flavin_anion_Pigeon_Resting_Z.npy", N)
    R_radical = load_R(r"C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\Final Results\SCT\flavin_anion_Pigeon_Radical_Z.npy", N)
    R_drosophila = load_R(r"C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\Final Results\SCT\flavin_anion_Drosophila_Z.npy", N)
    R_z = sct.lone_electron_R(np.array([0, 0, 0.05]), T)

    #singlet_prob_plot(1000, (R_no_bath, R_z), (R_resting, R_z), (R_radical, R_z), names=["No Bath", "Pigeon Resting", "Pigeon Radical"])

    coherence_plot(max(T)/gamma_e, (R_no_bath, R_z), (R_resting, R_z), (R_radical, R_z), (R_drosophila, R_z), names=["No Bath", "Pigeon Resting", "Pigeon Radical", "Drosophila"])



    # # plot singlet yields
    # path = r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data'
    # files = os.listdir(path)
    #
    # files = files[:12]
    # ks = np.logspace(np.log10(1 / (500 / 15)), 2, 100)
    # data = {file[:-4]: load_R(path+'\\'+file, N) for file in files}
    #
    # def field(direction):
    #     if direction == "X":
    #         return [0.05, 0, 0]
    #     if direction == "Y":
    #         return [0, 0.05, 0]
    #     if direction == "Z":
    #         return [0, 0, 0.05]
    #
    #
    # singlet_probs = {key: sct.singlet_prob(data[key], sct.lone_electron_R(field(key[-1]), T)) for key in data}
    # yield_data = {key: [sct.quantum_yield(k, singlet_probs[key], T / gamma_e) for k in ks] for key in singlet_probs}
    #
    #
    # fig = plt.figure(figsize=(12, 8))
    # for i in yield_data:
    #     print(i)
    #     plt.plot(np.log10(ks), yield_data[i])
    #
    # names = [key for key in yield_data]
    #
    # plt.legend(names)
    # plt.xlabel(r'log$_{10}$k ($\mu s^{-1}$)')
    # plt.ylabel(r'Singlet Yield')
    # plt.show()
    #
    # # Plot maximum difference in singlet yields
    # fig = plt.figure(figsize=(12, 8))
    # bath_names = []
    # for i in range(4):
    #     bath_names.append(names[i * 3][:-2])
    #     x = yield_data[names[i*3]]
    #     y = yield_data[names[(i * 3) + 1]]
    #     z = yield_data[names[(i * 3) + 2]]
    #     yield_min = np.min(np.vstack((x, y, z)), axis=0)
    #     yield_max = np.max(np.vstack((x, y, z)), axis=0)
    #     plt.plot(np.log10(ks), yield_max-yield_min)
    # bath_names = [i.replace('_', ' ') for i in bath_names]
    # bath_names = [i.split('flavin anion ')[-1] for i in bath_names]
    #
    # plt.legend(bath_names, bbox_to_anchor=(0.5, 1.10), loc='upper center', ncol=len(bath_names),
    #            shadow=True, fancybox=True, fontsize='xx-large')
    # plt.xlabel(r'log$_{10}$k ($\mu s^{-1}$)',  fontsize='xx-large')
    # plt.ylabel(r'Difference in $\Phi_s$', fontsize='xx-large')
    # plt.xticks(fontsize='x-large')
    # plt.yticks(fontsize='x-large')
    # plt.show()
    #
    # T_max = max(T) / gamma_e
    #
    # R_e_x = sct.lone_electron_R(field("X"), T)
    # R_e_y = sct.lone_electron_R(field("Y"), T)
    # R_e_z = sct.lone_electron_R(field("Z"), T)
    # R_r_x = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_No_Bath_X.npy', N)
    # R_r_y = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_No_Bath_Y.npy', N)
    # R_r_z = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_No_Bath_Z.npy', N)
    # singlet_prob_plot(T_max, (R_r_x, R_e_x), (R_r_y, R_e_y), (R_r_z, R_e_z),
    #                   names=("Field in X", "Field in Y", "Field in Z"), title="")
    #
    # N = 100
    # T = np.linspace(0, int(600 * 30 / N), int(600 * 170 / N))
    # R_e = sct.lone_electron_R([0,0,0.05], T)
    # R_dro = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_Drosophila_Z.npy', N)
    # R_no_bath = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_No_Bath_Z.npy', N)
    # R_pig_rest = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_Pigeon_Resting_Z.npy', N)
    # R_pig_rad = load_R(r'C:\Users\Kieran\PycharmProjects\magnetic-birds\Kieran\data\flavin_anion_Pigeon_Radical_Z.npy', N)
    #
    # coherence_plot(T_max, (R_dro, R_e), (R_no_bath, R_no_bath), (R_pig_rad, R_e), (R_pig_rest, R_e),
    #                names=("Drosophila", "No Bath", "Pigeon Radical", "Pigeon Resting"), title="")

