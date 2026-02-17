from qat.lang import Program, X, CNOT, RX, RY
from qat.core import Observable, Term
import library_measurments as lib
import numpy as np
from qat.qpus import PyLinalg
from qlmaas.qpus import LinAlg


def main():
    nbqbits = 4
    prog = Program()
    qbits = prog.qalloc(nbqbits)
    prog.apply(X, qbits[0])
    prog.apply(CNOT, qbits[0], qbits[2])
    for i, qb in enumerate(qbits):
        prog.apply(RX(0.324 * i), qb)

    circ = prog.to_circ()
    qpu = PyLinalg()
    # qpu = LinAlg() To try a different "QPU"
    direction = "Z"
    list_position = [3, 2]
    val_test = np.zeros(len(list_position))

    for i in range(len(list_position)):
        observable = Observable.z(list_position[i], nbqbits)
        job_obs = circ.to_job("OBS", observable=observable)
        result = qpu.submit(job_obs)
        val_test[i] = result.value

    print(val_test)
    val = lib.one_point_corr(circ, direction, list_position, qpu)
    print(val)

    direction = "Y"
    list_position = [[3, 2], [1, 2]]
    val_test = np.zeros(len(list_position))
    for i in range(len(list_position)):
        obs_y1 = Observable.y(list_position[i][0], nbqbits)
        obs_y2 = Observable.y(list_position[i][1], nbqbits)
        job_obs = circ.to_job("OBS", observable=obs_y1 * obs_y2)
        result = qpu.submit(job_obs)
        val_test[i] = result.value

    print(val_test)
    val = lib.two_point_corr(circ, direction, list_position, qpu)
    print(val)


if __name__ == "__main__":
    main()
