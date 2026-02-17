from qat.lang import Program, RX, RY
import numpy as np


def one_point_corr(circ, direction, list_position, qpu):
    """Compute one-point correlation function <sigma_i>.

    Computes the one-point correlation function <sigma_i> for each position
    in list_position, in the specified measurement basis. It is done by
    sampling the final state and accumulating probabilities weighted by the
    sign of the measured spin values.

    <sigma_i> = sum over samples of [ sign(bit_i) * probability ]

    where sign is -1 if the bit is 1 (spin down) and +1 if the bit is 0
    (spin up).

    Parameters:
      circ           Circuit    Input quantum circuit.
      direction      str        Measurement basis: 'X', 'Y', or 'Z'
                                'X' --> applies RY(-pi/2) to each qubit
                                'Y' --> applies RX(+pi/2) to each qubit
                                'Z' --> no gate applied (standard Z measure)
      list_position  list       List of qubit indices for which to compute
                                <sigma_i>. Each element is a single integer
                                index.
      qpu            QPU        Quantum processing unit used to run the
                                circuit.

    Returns:
      val            np.array   1D array of shape (len(list_position),)
                                val[i] = <sigma_{list_position[i]}>

    Example:
      # Compute <Z_0>, <Z_2>, <Z_4> on a 6-qubit state
      val = one_point_corr(circ, 'Z', [0, 2, 4], qpu)
    """
    prog_rot = Program()
    qbits = prog_rot.qalloc(circ.nbqbits)
    if direction == "X":
        for qb in qbits:
            prog_rot.apply(RY(-np.pi / 2), qb)
    elif direction == 'Y':
        for qb in qbits:
            prog_rot.apply(RX(np.pi / 2), qb)
    elif direction == 'Z':
        pass  # No rotation needed for Z basis measurement
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
        )

    circ_rot = prog_rot.to_circ()
    circ_final = circ + circ_rot
    job = circ_final.to_job()
    result_state = qpu.submit(job)
    val = np.zeros(len(list_position))
    for sample in result_state:
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        for i in range(len(list_position)):
            if bit_array[list_position[i]] == -1:
                val[i] += sample.probability
            else:
                val[i] -= sample.probability

    return val


def two_point_corr(circ, direction, list_position, qpu):
    """Compute two-point correlation function <sigma_i * sigma_j>.

    Computes the two-point correlation function <sigma_i * sigma_j> for each
    pair of positions in list_position, in the specified measurement basis.

    <sigma_i * sigma_j> = sum over samples of
                          [ sign(bit_i * bit_j) * probability ]

    The sign of the product is determined by np.prod() on the selected bits
    (already mapped to +1/-1). If the product is -1, the probability is
    subtracted; if +1, it is added.

    Parameters:
      prog_ext       Program    Input quantum program (will NOT be modified,
                                a deep copy is used internally).
      qbits          QRegister  Class for registers of qbits.
      direction      str        Measurement basis: 'X', 'Y', or 'Z'
                                'X' --> applies RY(-pi/2) to each qubit
                                'Y' --> applies RX(+pi/2) to each qubit
                                'Z' --> no gate applied (standard Z measure)
      list_position  list       List of pairs (or groups) of qubit indices.
                                Each element is a list of indices to multiply
                                together.
                                Example: [[0,1], [1,2], [2,3]] for
                                nearest-neighbor pairs.
      qpu            QPU        Quantum processing unit used to run the
                                circuit.

    Returns:
      val            np.array   1D array of shape (len(list_position),)
                                val[i] = <sigma_{list_position[i][0]} *
                                         sigma_{list_position[i][1]}>

    Example:
      # Compute <Z_0 Z_1>, <Z_1 Z_2>, <Z_2 Z_3>
      pairs = [[0, 1], [1, 2], [2, 3]]
      val = two_point_corr(circ, 'Z', pairs, qpu)
    """
    prog_rot = Program()
    qbits = prog_rot.qalloc(circ.nbqbits)
    if direction == "X":
        for qb in qbits:
            prog_rot.apply(RY(-np.pi / 2), qb)
    elif direction == 'Y':
        for qb in qbits:
            prog_rot.apply(RX(np.pi / 2), qb)

    circ_rot = prog_rot.to_circ()
    circ_final = circ + circ_rot
    job = circ_final.to_job()
    result_state = qpu.submit(job)
    val = np.zeros(len(list_position))
    for sample in result_state:
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        for i in range(len(list_position)):
            if np.prod(bit_array[list_position[i]]) == -1:
                val[i] -= sample.probability
            else:
                val[i] += sample.probability

    return val
