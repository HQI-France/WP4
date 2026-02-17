import library_annealing as lib
from qlmaas.qpus import AnalogQPU, MPSTraj


def main():
    nb_spins = 4
    h_target = -0.8
    J_target = -1
    tau = 10
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))
    a = lib.quantum_annealing_ising(nb_spins, h_target, J_target, periodic,
                                     tau, qpu)
    print(a)


if __name__ == "__main__":
    main()
