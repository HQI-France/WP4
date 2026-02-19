
import copy
import numpy as np
from qat.lang import Program, RX, RY
from qat.core import Term, Observable, Schedule, Batch


def one_point_corr_circ(circ, direction, list_position, qpu):
    """Compute one-point correlation function <sigma_i> from a circuit.

    Computes the one-point correlation function <sigma_i> for each position
    in list_position, in the specified measurement basis. It is done by
    sampling the final state and accumulating probabilities weighted by the
    sign of the measured spin values.

    <sigma_i> = sum over samples of [ sign(bit_i) * probability ]

    where sign is -1 if the bit is 1 (spin down) and +1 if the bit is 0
    (spin up).

    Parameters
    ----------
    circ : Circuit
        Input quantum circuit (qat.core.Circuit)
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> applies RY(-pi/2) to each qubit
        'Y' --> applies RX(+pi/2) to each qubit
        'Z' --> no gate applied (standard Z measurement)
    list_position : list of int
        List of qubit indices for which to compute <sigma_i>
        Each element is a single integer index
    qpu : QPU
        Quantum processing unit used to run the circuit

    Returns
    -------
    val : np.ndarray
        1D array of shape (len(list_position),)
        val[i] = <sigma_{list_position[i]}>

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0>, <Z_2>, <Z_4> on a 6-qubit state
    >>> val = one_point_corr_circ(circ, 'Z', [0, 2, 4], qpu)
    """
    # Build rotation circuit for basis change
    prog_rot = Program()
    qbits = prog_rot.qalloc(circ.nbqbits)
    
    if direction == "X":
        # Rotate X basis into Z basis
        for qb in qbits:
            prog_rot.apply(RY(-np.pi / 2), qb)
    elif direction == 'Y':
        # Rotate Y basis into Z basis
        for qb in qbits:
            prog_rot.apply(RX(np.pi / 2), qb)
    elif direction == 'Z':
        pass  # No rotation needed for Z basis measurement
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
        )

    # Combine original circuit with rotation gates
    circ_rot = prog_rot.to_circ()
    circ_final = circ + circ_rot
    
    # Submit job and sample final state
    job = circ_final.to_job()
    result_state = qpu.submit(job)
    
    # Accumulate correlation values
    val = np.zeros(len(list_position))
    for sample in result_state:
        # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        
        # Compute <sigma_i> for each position
        for i in range(len(list_position)):
            if bit_array[list_position[i]] == -1:
                val[i] += sample.probability
            else:
                val[i] -= sample.probability

    return val


def two_point_corr_circ(circ, direction, list_position, qpu):
    """Compute two-point correlation function <sigma_i * sigma_j> from a circuit.

    Computes the two-point correlation function <sigma_i * sigma_j> for each
    pair of positions in list_position, in the specified measurement basis.

    <sigma_i * sigma_j> = sum over samples of
                          [ sign(bit_i * bit_j) * probability ]

    The sign of the product is determined by np.prod() on the selected bits
    (already mapped to +1/-1). If the product is -1, the probability is
    subtracted; if +1, it is added.

    Parameters
    ----------
    circ : Circuit
        Input quantum circuit (qat.core.Circuit)
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> applies RY(-pi/2) to each qubit
        'Y' --> applies RX(+pi/2) to each qubit
        'Z' --> no gate applied (standard Z measurement)
    list_position : list of list of int
        List of pairs (or groups) of qubit indices
        Each element is a list of indices to multiply together
        Example: [[0,1], [1,2], [2,3]] for nearest-neighbor pairs
    qpu : QPU
        Quantum processing unit used to run the circuit

    Returns
    -------
    val : np.ndarray
        1D array of shape (len(list_position),)
        val[i] = <sigma_{list_position[i][0]} * sigma_{list_position[i][1]}>

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0 Z_1>, <Z_1 Z_2>, <Z_2 Z_3>
    >>> pairs = [[0, 1], [1, 2], [2, 3]]
    >>> val = two_point_corr_circ(circ, 'Z', pairs, qpu)
    """
    # Build rotation circuit for basis change
    prog_rot = Program()
    qbits = prog_rot.qalloc(circ.nbqbits)
    
    if direction == "X":
        # Rotate X basis into Z basis
        for qb in qbits:
            prog_rot.apply(RY(-np.pi / 2), qb)
    elif direction == 'Y':
        # Rotate Y basis into Z basis
        for qb in qbits:
            prog_rot.apply(RX(np.pi / 2), qb)
    elif direction == 'Z':
        pass  # No rotation needed for Z basis measurement
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
        )

    # Combine original circuit with rotation gates
    circ_rot = prog_rot.to_circ()
    circ_final = circ + circ_rot
    
    # Submit job and sample final state
    job = circ_final.to_job()
    result_state = qpu.submit(job)
    
    # Accumulate correlation values
    val = np.zeros(len(list_position))
    for sample in result_state:
        # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        
        # Compute <sigma_i * sigma_j * ...> for each group
        for i in range(len(list_position)):
            if np.prod(bit_array[list_position[i]]) == -1:
                val[i] -= sample.probability
            else:
                val[i] += sample.probability

    return val


def sched_rot(angle, axes, n):
    """Create a rotation schedule applied uniformly to all n qubits.
    
    Used to prepare the initial state or perform basis changes in
    schedule-based calculations.

    Parameters
    ----------
    angle : float
        Rotation angle in radians (e.g. -pi/4)
    axes : str
        Pauli axis for rotation ('X', 'Y', or 'Z')
    n : int
        Number of qubits

    Returns
    -------
    Schedule
        qat.core.Schedule object with tmax=1
    """
    # Create rotation terms for all qubits
    rot = [Term(angle, axes, [i]) for i in range(n)]
    
    # Build the Hamiltonian from rotation terms
    rot_ham = Observable(n, pauli_terms=rot)
    
    # Create the schedule with unit time
    drive_rot = [(1, rot_ham)]
    schedule_rot = Schedule(drive=drive_rot, tmax=1)
    
    return schedule_rot


def one_point_corr(sched, direction, list_position, qpu, initial_state=None):
    """Compute one-point correlation function <sigma_i> from a schedule.

    Computes the one-point correlation function <sigma_i> for each position
    in list_position, in the specified measurement basis. It is done by
    sampling the final state and accumulating probabilities weighted by the
    sign of the measured spin values.

    <sigma_i> = sum over samples of [ sign(bit_i) * probability ]

    where sign is -1 if the bit is 1 (spin down) and +1 if the bit is 0
    (spin up).

    Parameters
    ----------
    sched : Schedule
        Input quantum schedule (qat.core.Schedule)
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> appends rotation schedule to measure X
        'Y' --> appends rotation schedule to measure Y
        'Z' --> no rotation (standard Z measurement)
    list_position : list of int
        List of qubit indices for which to compute <sigma_i>
        Each element is a single integer index
    qpu : QPU
        Quantum processing unit used to run the schedule
    initial_state : np.ndarray or str, optional
        Initial state for the schedule (default: None)

    Returns
    -------
    val : np.ndarray
        1D array of shape (len(list_position),)
        val[i] = <sigma_{list_position[i]}>

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0>, <Z_2>, <Z_4> from a schedule
    >>> val = one_point_corr(sched, 'Z', [0, 2, 4], qpu)
    """
    # Get number of qubits from schedule
    nbqbits = sched.nbqbits
    
    # Build final schedule with basis rotation if needed
    if direction == "X":
        # Rotate from Z to X basis
        sched_final = sched | sched_rot(-np.pi / 4, "Y", nbqbits)
    elif direction == 'Y':
        # Rotate from Z to Y basis
        sched_final = sched | sched_rot(np.pi / 4, "X", nbqbits)
    elif direction == 'Z':
        # Deep copy to avoid modifying original schedule
        sched_final = copy.deepcopy(sched)
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
        )

    # Submit job with initial state
    job = sched_final.to_job(psi_0=initial_state)
    result_state = qpu.submit(job)
    
    # Accumulate correlation values
    val = np.zeros(len(list_position))
    for sample in result_state:
        # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        
        # Compute <sigma_i> for each position
        for i in range(len(list_position)):
            if bit_array[list_position[i]] == -1:
                val[i] += sample.probability
            else:
                val[i] -= sample.probability

    return val


def two_point_corr(sched, direction, list_position, qpu, initial_state=None):
    """Compute two-point correlation function <sigma_i * sigma_j> from a schedule.

    Computes the two-point correlation function <sigma_i * sigma_j> for each
    pair of positions in list_position, in the specified measurement basis.

    <sigma_i * sigma_j> = sum over samples of
                          [ sign(bit_i * bit_j) * probability ]

    The sign of the product is determined by np.prod() on the selected bits
    (already mapped to +1/-1). If the product is -1, the probability is
    subtracted; if +1, it is added.

    Parameters
    ----------
    sched : Schedule
        Input quantum schedule (qat.core.Schedule)
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> appends rotation schedule to measure X
        'Y' --> appends rotation schedule to measure Y
        'Z' --> no rotation (standard Z measurement)
    list_position : list of list of int
        List of pairs (or groups) of qubit indices
        Each element is a list of indices to multiply together
        Example: [[0,1], [1,2], [2,3]] for nearest-neighbor pairs
    qpu : QPU
        Quantum processing unit used to run the schedule
    initial_state : np.ndarray or str, optional
        Initial state for the schedule (default: None)

    Returns
    -------
    val : np.ndarray
        1D array of shape (len(list_position),)
        val[i] = <sigma_{list_position[i][0]} * sigma_{list_position[i][1]}>

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0 Z_1>, <Z_1 Z_2>, <Z_2 Z_3> from a schedule
    >>> pairs = [[0, 1], [1, 2], [2, 3]]
    >>> val = two_point_corr(sched, 'Z', pairs, qpu)
    """
    # Get number of qubits from schedule
    nbqbits = sched.nbqbits
    
    # Build final schedule with basis rotation if needed
    if direction == "X":
        # Rotate from Z to X basis
        sched_final = sched | sched_rot(-np.pi / 4, "Y", nbqbits)
    elif direction == 'Y':
        # Rotate from Z to Y basis
        sched_final = sched | sched_rot(np.pi / 4, "X", nbqbits)
    elif direction == 'Z':
        # Deep copy to avoid modifying original schedule
        sched_final = copy.deepcopy(sched)
    else:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
        )

    # Submit job with initial state
    job = sched_final.to_job(psi_0=initial_state)
    result_state = qpu.submit(job)
    
    # Accumulate correlation values
    val = np.zeros(len(list_position))
    for sample in result_state:
        # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
        bit_array = np.array(list(sample.state.bitstring), dtype=int)
        bit_array[bit_array == 0] = -1
        
        # Compute <sigma_i * sigma_j * ...> for each group
        for i in range(len(list_position)):
            if np.prod(bit_array[list_position[i]]) == -1:
                val[i] -= sample.probability
            else:
                val[i] += sample.probability

    return val


def one_point_corr_batch(
    list_sched, direction, list_position, qpu, initial_state=None
):
    """Compute one-point correlation function <sigma_i> for a batch of schedules.

    Computes the one-point correlation function <sigma_i> for each position
    in list_position, in the specified measurement basis. It is done by
    sampling the final state and accumulating probabilities weighted by the
    sign of the measured spin values.

    <sigma_i> = sum over samples of [ sign(bit_i) * probability ]

    where sign is -1 if the bit is 1 (spin down) and +1 if the bit is 0
    (spin up).

    Parameters
    ----------
    list_sched : list of Schedule
        List of quantum schedules (qat.core.Schedule) to process in batch
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> appends rotation schedule to measure X
        'Y' --> appends rotation schedule to measure Y
        'Z' --> no rotation (standard Z measurement)
    list_position : list of int
        List of qubit indices for which to compute <sigma_i>
        Each element is a single integer index
    qpu : QPU
        Quantum processing unit used to run the batch
    initial_state : list of (np.ndarray or str), optional
        List of initial states for each schedule (default: None)
        If None, uses default initial state for all schedules

    Returns
    -------
    val : np.ndarray
        2D array of shape (len(list_sched), len(list_position))
        val[j, i] = <sigma_{list_position[i]}> for schedule j

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0>, <Z_2>, <Z_4> for multiple schedules
    >>> val = one_point_corr_batch(schedules, 'Z', [0, 2, 4], qpu)
    """
    # Build list of jobs for batch submission
    list_jobs = []

    for i in range(len(list_sched)):
        # Get number of qubits from this schedule
        nbqbits = list_sched[i].nbqbits
        
        # Build final schedule with basis rotation if needed
        if direction == "X":
            # Rotate from Z to X basis
            sched_final = list_sched[i] | sched_rot(-np.pi / 4, "Y", nbqbits)
        elif direction == 'Y':
            # Rotate from Z to Y basis
            sched_final = list_sched[i] | sched_rot(np.pi / 4, "X", nbqbits)
        elif direction == 'Z':
            # Deep copy to avoid modifying original schedule
            sched_final = copy.deepcopy(list_sched[i])
        else:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
            )
        
        # Create job with or without initial state
        if initial_state == None:
            list_jobs.append(sched_final.to_job())
        else:
            list_jobs.append(sched_final.to_job(psi_0=initial_state[i]))
    
    # Submit batch and get results
    batch = Batch(list_jobs)
    batch_res = qpu.submit(batch)
    
    # Initialize result array
    val = np.zeros((len(list_sched), len(list_position)))
    
    # Process each schedule result
    for j in range(len(list_sched)):
        for sample in batch_res[j]:
            # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
            bit_array = np.array(list(sample.state.bitstring), dtype=int)
            bit_array[bit_array == 0] = -1
            
            # Compute <sigma_i> for each position
            for i in range(len(list_position)):
                if bit_array[list_position[i]] == -1:
                    val[j, i] += sample.probability
                else:
                    val[j, i] -= sample.probability
                
    return val


def two_point_corr_batch(
    list_sched, direction, list_position, qpu, initial_state=None
):
    """Compute two-point correlation function <sigma_i * sigma_j> for a batch of schedules.

    Computes the two-point correlation function <sigma_i * sigma_j> for each
    pair of positions in list_position, in the specified measurement basis.

    <sigma_i * sigma_j> = sum over samples of
                          [ sign(bit_i * bit_j) * probability ]

    The sign of the product is determined by np.prod() on the selected bits
    (already mapped to +1/-1). If the product is -1, the probability is
    subtracted; if +1, it is added.

    Parameters
    ----------
    list_sched : list of Schedule
        List of quantum schedules (qat.core.Schedule) to process in batch
    direction : str
        Measurement basis: 'X', 'Y', or 'Z'
        'X' --> appends rotation schedule to measure X
        'Y' --> appends rotation schedule to measure Y
        'Z' --> no rotation (standard Z measurement)
    list_position : list of list of int
        List of pairs (or groups) of qubit indices
        Each element is a list of indices to multiply together
        Example: [[0,1], [1,2], [2,3]] for nearest-neighbor pairs
    qpu : QPU
        Quantum processing unit used to run the batch
    initial_state : list of (np.ndarray or str), optional
        List of initial states for each schedule (default: None)
        If None, uses default initial state for all schedules

    Returns
    -------
    val : np.ndarray
        2D array of shape (len(list_sched), len(list_position))
        val[j, i] = <sigma_{list_position[i][0]} * sigma_{list_position[i][1]}>
        for schedule j

    Raises
    ------
    ValueError
        If direction is not 'X', 'Y', or 'Z'

    Example
    -------
    >>> # Compute <Z_0 Z_1>, <Z_1 Z_2>, <Z_2 Z_3> for multiple schedules
    >>> pairs = [[0, 1], [1, 2], [2, 3]]
    >>> val = two_point_corr_batch(schedules, 'Z', pairs, qpu)
    """
    # Build list of jobs for batch submission
    list_jobs = []

    for i in range(len(list_sched)):
        # Get number of qubits from this schedule
        nbqbits = list_sched[i].nbqbits
        
        # Build final schedule with basis rotation if needed
        if direction == "X":
            # Rotate from Z to X basis
            sched_final = list_sched[i] | sched_rot(-np.pi / 4, "Y", nbqbits)
        elif direction == 'Y':
            # Rotate from Z to Y basis
            sched_final = list_sched[i] | sched_rot(np.pi / 4, "X", nbqbits)
        elif direction == 'Z':
            # Deep copy to avoid modifying original schedule
            sched_final = copy.deepcopy(list_sched[i])
        else:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'X', 'Y', or 'Z'."
            )
        
        # Create job with or without initial state
        if initial_state == None:
            list_jobs.append(sched_final.to_job())
        else:
            list_jobs.append(sched_final.to_job(psi_0=initial_state[i]))
    
    # Submit batch and get results
    batch = Batch(list_jobs)
    batch_res = qpu.submit(batch)
    
    # Initialize result array
    val = np.zeros((len(list_sched), len(list_position)))
    
    # Process each schedule result
    for j in range(len(list_sched)):
        for sample in batch_res[j]:
            # Convert bitstring to array and map 0->-1, 1->1 (Ising convention)
            bit_array = np.array(list(sample.state.bitstring), dtype=int)
            bit_array[bit_array == 0] = -1
            
            # Compute <sigma_i * sigma_j * ...> for each group
            for i in range(len(list_position)):
                if np.prod(bit_array[list_position[i]]) == -1:
                    val[j, i] -= sample.probability
                else:
                    val[j, i] += sample.probability
                
    return val
