"""Quantum annealing library for Ising model simulation."""

import numpy as np
from qat.core import Observable, Term, Schedule, Batch
from qat.core.variables import Variable


def write_ising_ham(n, h, J, periodic):
    """Build operators, positions, and coefficients for an Ising Hamiltonian.
    
    H = h * sum(Z_i) + J * sum(XX_{i,i+1})

    Parameters
    ----------
    n : int
        Number of spins
    h : float or Variable
        Transverse field coefficient (can be a Variable for scheduling)
    J : float or Variable
        Coupling coefficient (can be a Variable for scheduling)
    periodic : bool
        If True, adds a coupling between the last and first spin

    Returns
    -------
    operator : list of str
        Pauli operator strings (e.g. 'Z', 'XX')
    position : list of list of int
        Qubit positions for each term
    coeff : list of float or Variable
        Coefficients for each term
    """
    # Initialize arrays with placeholders
    operator = ['0' for i in range((2 * n - 1))]
    operator[:n] = ['Z' for i in range(n)]  # Single-qubit Z terms
    operator[n:] = ['XX' for i in range(n, 2 * n - 1)]  # Two-qubit XX terms

    # Qubit indices for each operator
    position = [[0] for i in range(2 * n - 1)]
    position[:n] = [[i] for i in range(n)]  # Z acts on single qubits
    position[n:] = [[i, i + 1] for i in range(n - 1)]  # XX on neighbors

    # Coefficients: h for Z terms, J for XX terms
    coeff = [[0] for i in range(2 * n - 1)]
    coeff[:n] = [h for i in range(n)]
    coeff[n:] = [J for i in range(n, 2 * n - 1)]

    # Add periodic boundary condition if requested
    if periodic and n > 2:
        operator.append('XX')
        position.append([n - 1, 0])  # Connect last spin to first
        coeff.append(J)

    return operator, position, coeff


def sched_rot(angle, axes, n):
    """Create a rotation schedule applied uniformly to all n qubits.
    
    Used to prepare the initial state before annealing, starting from the
    ferromagnetic side.

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


def sched_go_ferro(nb_spins, h_target, J_target, periodic, tau):
    """Create forward schedule for ferromagnetic annealing.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Target value for transverse field h
    J_target : float
        Ferromagnetic coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Time scale parameter controlling annealing speed
        
    Returns
    -------
    Schedule
        qat.core.Schedule object for the forward ramp
    """

    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    
    if np.abs(h_target) > np.abs(J_target):
        raise ValueError(
            "This is not the schedule that you are looking for.\n"
            "There are paramagnetic parameters."
        )
    
    # Time variable for the schedule
    t = Variable("t", float)

    # Ramp h from 0 to h_target linearly
    h_go = np.sign(h_target) * t / tau
    time_go = np.abs(h_target) * tau
    
    # Build the Ising Hamiltonian with time-dependent h
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(
        nb_spins, h_go, J_target, periodic
    )
    hamiltonian = [
        Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i])
        for i in range(len(oper_ham))
    ]
    ham = Observable(nb_spins, pauli_terms=hamiltonian)
    drive = [(1, ham)]
    schedule_go = Schedule(drive=drive, tmax=time_go)
    
    return schedule_go


def sched_back_ferro(nb_spins, h_target, J_target, periodic, tau):
    """Create backward schedule for ferromagnetic annealing.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Target value for transverse field h
    J_target : float
        Ferromagnetic coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Time scale parameter controlling annealing speed
        
    Returns
    -------
    Schedule
        qat.core.Schedule object for the backward ramp
    """
    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    
    if np.abs(h_target) > np.abs(J_target):
        raise ValueError(
            "This is not the schedule that you are looking for.\n"
            "There are paramagnetic parameters."
        )
    
    # Time variable for the schedule
    t = Variable("t", float)

    # Ramp h from h_target back down to 0 linearly
    h_back = h_target - np.sign(h_target) * t / tau
    time_back = np.abs(h_target) * tau
    
    # Build the Ising Hamiltonian with time-dependent h
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(
        nb_spins, h_back, J_target, periodic
    )
    hamiltonian = [
        Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i])
        for i in range(len(oper_ham))
    ]
    ham = Observable(nb_spins, pauli_terms=hamiltonian)
    drive = [(1, ham)]
    schedule_back = Schedule(drive=drive, tmax=time_back)

    return schedule_back


def sched_go_para(nb_spins, h_target, J_target, periodic, tau):
    """Create forward schedule for paramagnetic annealing.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Transverse field value (kept constant during annealing)
    J_target : float
        Target coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Time scale parameter controlling annealing speed
        
    Returns
    -------
    Schedule
        qat.core.Schedule object for the forward ramp
    """
    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    
    if np.abs(h_target) < np.abs(J_target):
        raise ValueError(
            "This is not the schedule that you are looking for.\n"
            "There are ferromagnetic parameters."
        )

    # Time variable for the schedule
    t = Variable("t", float)

    # Ramp J from 0 to J_target linearly (J is negative)
    j_go = -t / tau
    time_go = np.abs(J_target * tau)
    
    # Build the Ising Hamiltonian with time-dependent J
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(
        nb_spins, h_target, j_go, periodic
    )
    hamiltonian = [
        Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i])
        for i in range(len(oper_ham))
    ]
    ham = Observable(nb_spins, pauli_terms=hamiltonian)
    drive = [(1, ham)]
    schedule_go = Schedule(drive=drive, tmax=time_go)

    return schedule_go


def sched_back_para(nb_spins, h_target, J_target, periodic, tau):
    """Create backward schedule for paramagnetic annealing.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Transverse field value (kept constant during annealing)
    J_target : float
        Target coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Time scale parameter controlling annealing speed
        
    Returns
    -------
    Schedule
        qat.core.Schedule object for the backward ramp
    """

    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    
    if np.abs(h_target) < np.abs(J_target):
        raise ValueError(
            "This is not the schedule that you are looking for.\n"
            "There are ferromagnetic parameters."
        )
    # Time variable for the schedule
    t = Variable("t", float)

    # Ramp J from J_target back down to 0 linearly
    j_back = J_target + t / tau
    time_back = np.abs(J_target * tau)
    
    # Build the Ising Hamiltonian with time-dependent J
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(
        nb_spins, h_target, j_back, periodic
    )
    hamiltonian = [
        Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i])
        for i in range(len(oper_ham))
    ]
    ham = Observable(nb_spins, pauli_terms=hamiltonian)
    drive = [(1, ham)]
    schedule_back = Schedule(drive=drive, tmax=time_back)

    return schedule_back


def density_def_ferro(n, periodic):
    """Define the ferromagnetic kink operator.
    
    Counts the number of neighbouring spins not aligned:
    rho = (1/N) * sum(XX_{i,i+1})
    
    Used to measure how close the system is to the ferromagnetic ground state.

    Parameters
    ----------
    n : int
        Number of spins
    periodic : bool
        If True, includes the XX term between last and first spin

    Returns
    -------
    Observable
        Observable (qat.core)
    """
    operator = ['XX' for i in range(n - 1)]
    position = [[i, i + 1] for i in range(n - 1)]

    if periodic and n > 2:
        operator.append('XX')
        position.append([n - 1, 0])

    coeff = [1 / len(operator) for i in range(len(operator))]
    density = [
        Term(coeff[i], operator[i], position[i])
        for i in range(len(operator))
    ]
    dens = Observable(n, pauli_terms=density)
    return dens


def density_def_para(n):
    """Define the paramagnetic kink operator.
    
    Counts the number of spins aligned in the opposite direction of the
    transverse magnetic field: rho = (1/N) * sum(Z_{i})
    
    Used to measure how close the system is to the paramagnetic ground state.

    Parameters
    ----------
    n : int
        Number of spins

    Returns
    -------
    Observable
        Observable (qat.core)
    """
    operator = ['Z' for i in range(n)]
    position = [[i] for i in range(n)]

    coeff = [1 / len(operator) for i in range(len(operator))]
    density = [
        Term(coeff[i], operator[i], position[i])
        for i in range(len(operator))
    ]
    dens = Observable(n, pauli_terms=density)
    return dens


def estimate_annealing_ferro(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Run a ferromagnetic quantum annealing protocol.
    
    The ramp back down is used to measure the defects generated in a known
    basis, so if we do not accumulate defects in the final state it means
    that we were close to the instantaneous ground state all along the time
    evolution.

    The schedule has 3 phases:
    1. Initial rotation by -pi/4 around Y axis
    2. Ramp up h from 0 to h_target over time |h_target| * tau
    3. Ramp h back down to 0 over time |h_target| * tau

    The initial state is the symmetric/antisymmetric superposition:
    gs[0] = 1/sqrt(2), gs[-1] = +/-1/sqrt(2) (sign depends on parity of n)

    Parameters
    ----------
    nb_spins : int
        Number of spins
    h_target : float
        Target transverse field value
    J_target : float
        Ferromagnetic coupling (should be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter
    qpu : QPU
        Quantum processing unit to run the job

    Returns
    -------
    float
        Defect density = (1 - <rho_ferro>) / 2
    """
    # Build the three-phase schedule: rotation -> forward -> backward
    schedule_rot = sched_rot(-np.pi / 4, "Y", nb_spins)
    schedule_go = sched_go_ferro(nb_spins, h_target, J_target, periodic, tau)
    schedule_back = sched_back_ferro(
        nb_spins, h_target, J_target, periodic, tau
    )
    sched_fin = schedule_rot | schedule_go | schedule_back

    # Prepare initial ground state (symmetric or antisymmetric superposition)
    gs = np.zeros(2**nb_spins)
    if nb_spins % 2 == 0:
        gs[0] = 1 / np.sqrt(2)
        gs[-1] = 1 / np.sqrt(2)
    else:
        gs[0] = 1 / np.sqrt(2)
        gs[-1] = -1 / np.sqrt(2)

    # Submit job and measure ferromagnetic order parameter
    job_sample = sched_fin.to_job(
        psi_0=gs, observable=density_def_ferro(nb_spins, periodic)
    )
    res_sample = qpu.submit(job_sample)
    job_num = res_sample.job_id
    res_sample.join()

    # Return defect density
    return (1 - res_sample.value) / 2


def estimate_annealing_ferro_batch(
    nb_spins, list_h_target, list_J_target, periodic, tau, qpu
):
    """Run a batch of ferromagnetic quantum annealing protocols.
    
    The ramp back down is used to measure the defects generated in a known
    basis, so if we do not accumulate defects in the final state it means
    that we were close to the instantaneous ground state all along the time
    evolution.

    The schedule has 3 phases:
    1. Initial rotation by -pi/4 around Y axis
    2. Ramp up h from 0 to h_target over time |h_target| * tau
    3. Ramp h back down to 0 over time |h_target| * tau

    The initial state is the symmetric/antisymmetric superposition:
    gs[0] = 1/sqrt(2), gs[-1] = +/-1/sqrt(2) (sign depends on parity of n)

    Parameters
    ----------
    nb_spins : int
        Number of spins
    list_h_target : array_like
        Array of target transverse field values
    list_J_target : array_like
        Array of ferromagnetic coupling values (should be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter
    qpu : QPU
        Quantum processing unit to run the batch job

    Returns
    -------
    density : np.ndarray
        Array of defect densities = (1 - <rho_ferro>) / 2 for each parameter set
    """
    # Build list of jobs for batch submission
    list_jobs = []
    for i in range(len(list_h_target)):
        # Build the three-phase schedule for this parameter set
        schedule_rot = sched_rot(-np.pi / 4, "Y", nb_spins)
        schedule_go = sched_go_ferro(
            nb_spins, list_h_target[i], list_J_target[i], periodic, tau
        )
        schedule_back = sched_back_ferro(
            nb_spins, list_h_target[i], list_J_target[i], periodic, tau
        )
        sched_fin = schedule_rot | schedule_go | schedule_back

        # Prepare initial ground state
        gs = np.zeros(2**nb_spins)
        if nb_spins % 2 == 0:
            gs[0] = 1 / np.sqrt(2)
            gs[-1] = 1 / np.sqrt(2)
        else:
            gs[0] = 1 / np.sqrt(2)
            gs[-1] = -1 / np.sqrt(2)

        # Create job with observable
        job_sample = sched_fin.to_job(
            psi_0=gs, observable=density_def_ferro(nb_spins, periodic)
        )
        list_jobs.append(job_sample)

    # Submit batch and wait for results
    batch_sample = Batch(list_jobs)
    batch_res = qpu.submit(batch_sample)
    batch_res.join()
    
    # Extract defect densities from batch results
    density = np.zeros(len(list_h_target))
    for i in range(len(batch_res)):
        density[i] = (1 - batch_res[i].value) / 2

    return density

def estimate_annealing_para(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Run a paramagnetic quantum annealing protocol.
    
    The ramp back down is used to measure the defects generated in a known
    basis, so if we do not accumulate defects in the final state it means
    that we were close to the instantaneous ground state all along the time
    evolution.

    The schedule has 2 phases:
    1. Ramp up |J| from 0 to J_target over time |J_target| * tau
    2. Ramp J back down to 0 over time |J_target| * tau

    Initial state:
    '000...0' if h_target > 0 (all spins up)
    '111...1' if h_target < 0 (all spins down)

    Parameters
    ----------
    nb_spins : int
        Number of spins
    h_target : float
        Transverse field value (constant throughout)
    J_target : float
        Target coupling value (should be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter
    qpu : QPU
        Quantum processing unit to run the job

    Returns
    -------
    float
        Defect density:
        1 - <rho_para>  if h_target > 0
        1 + <rho_para>  if h_target < 0
    """
    schedule_go = sched_go_para(nb_spins, h_target, J_target, periodic, tau)
    schedule_back = sched_back_para(
        nb_spins, h_target, J_target, periodic, tau
    )

    sched_fin = schedule_go | schedule_back
    if h_target > 0:
        psi_in = '0' * nb_spins
    else:
        psi_in = '1' * nb_spins

    job_sample = sched_fin.to_job(
        psi_0=psi_in, observable=density_def_para(nb_spins)
    )
    res_sample = qpu.submit(job_sample)
    job_num = res_sample.job_id
    res_sample.join()

    if h_target > 0:
        result = 1 - res_sample.value
    else:
        result = 1 + res_sample.value

    return result


def estimate_annealing_para_batch(
    nb_spins, list_h_target, list_J_target, periodic, tau, qpu
):
    """Run a batch of paramagnetic quantum annealing protocols.
    
    The ramp back down is used to measure the defects generated in a known
    basis, so if we do not accumulate defects in the final state it means
    that we were close to the instantaneous ground state all along the time
    evolution.

    The schedule has 2 phases:
    1. Ramp up |J| from 0 to J_target over time |J_target| * tau
    2. Ramp J back down to 0 over time |J_target| * tau

    Initial state:
    '000...0' if h_target > 0 (all spins up)
    '111...1' if h_target < 0 (all spins down)

    Parameters
    ----------
    nb_spins : int
        Number of spins
    list_h_target : array_like
        Array of transverse field values (constant throughout)
    list_J_target : array_like
        Array of target coupling values (should be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter
    qpu : QPU
        Quantum processing unit to run the batch job

    Returns
    -------
    density : np.ndarray
        Array of defect densities:
        1 - <rho_para>  if h_target > 0
        1 + <rho_para>  if h_target < 0
    """
    # Build list of jobs for batch submission
    list_jobs = []
    for i in range(len(list_h_target)):
        # Build the two-phase schedule for this parameter set
        schedule_go = sched_go_para(
            nb_spins, list_h_target[i], list_J_target[i], periodic, tau
        )
        schedule_back = sched_back_para(
            nb_spins, list_h_target[i], list_J_target[i], periodic, tau
        )

        sched_fin = schedule_go | schedule_back
        
        # Prepare initial state aligned with transverse field
        if list_h_target[i] > 0:
            psi_in = '0' * nb_spins  # All spins up
        else:
            psi_in = '1' * nb_spins  # All spins down

        # Create job with observable
        job_sample = sched_fin.to_job(
            psi_0=psi_in, observable=density_def_para(nb_spins)
        )
        list_jobs.append(job_sample)

    # Submit batch and wait for results
    batch_sample = Batch(list_jobs)
    batch_res = qpu.submit(batch_sample)
    batch_res.join()
    
    # Extract defect densities from batch results
    density = np.zeros(len(list_h_target))
    for i in range(len(batch_res)):
        if list_h_target[i] > 0:
            density[i] = 1 - batch_res[i].value
        else:
            density[i] = 1 + batch_res[i].value

    return density


def estimate_state_quality(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Automatically select the correct annealing protocol.
    
    Based on the relative magnitudes of h_target and J_target. Bear in mind
    that since we are also going back to the initial state if we choose a tau
    too small we would get very few defects (The state does not change), still
    it is not a reliable number. Tau should be at least of the order of unity.

    |h| > |J|  -->  paramagnetic protocol (quantum_annealing_para)
    |h| < |J|  -->  ferromagnetic protocol (quantum_annealing_ferro)

    Note: Anti-ferromagnetic coupling (J > 0) is not supported.
          The function will raise an error if J_target > 0.

    Parameters
    ----------
    nb_spins : int
        Number of spins
    h_target : float
        Transverse field target value
    J_target : float
        Coupling target value (must be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter (larger = slower = fewer defects)
    qpu : QPU
        Quantum processing unit to run the job

    Returns
    -------
    float
        Defect density (between 0 and 1)
    """
    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    elif tau <= 1:
        raise ValueError(
            "This protocol is not valid for fast protocol. "
            "Please choose tau > 1."
        )
    else:
        if np.abs(h_target) > np.abs(J_target):
            defects = estimate_annealing_para(
                nb_spins, h_target, J_target, periodic, tau, qpu
            )
        else:
            defects = estimate_annealing_ferro(
                nb_spins, h_target, J_target, periodic, tau, qpu
            )

    return defects


def estimate_state_quality_batch(
    nb_spins, list_h_target, list_J_target, periodic, tau, qpu
):
    """Automatically select and run annealing protocols in batch mode.
    
    Based on the relative magnitudes of h_target and J_target for each
    parameter set. Bear in mind that since we are also going back to the
    initial state if we choose a tau too small we would get very few defects
    (The state does not change), still it is not a reliable number. Tau should
    be at least of the order of unity.

    |h| > |J|  -->  paramagnetic protocol (estimate_annealing_para_batch)
    |h| < |J|  -->  ferromagnetic protocol (estimate_annealing_ferro_batch)

    Note: Anti-ferromagnetic coupling (J > 0) is not supported.
          The function will raise an error if any J_target > 0.

    Parameters
    ----------
    nb_spins : int
        Number of spins
    list_h_target : array_like
        Array of transverse field target values
    list_J_target : array_like
        Array of coupling target values (must all be < 0)
    periodic : bool
        Periodic boundary conditions
    tau : float
        Annealing speed parameter (larger = slower = fewer defects, must be > 1)
    qpu : QPU
        Quantum processing unit to run the batch jobs

    Returns
    -------
    density : np.ndarray
        Array of defect densities (between 0 and 1) for each parameter set
        
    Raises
    ------
    ValueError
        If any J_target > 0 (anti-ferromagnetic) or tau <= 1
    """
    # Convert inputs to numpy arrays for vectorized operations
    list_h_target = np.array(list_h_target)
    list_J_target = np.array(list_J_target)
    
    # Validate that all J values are negative (ferromagnetic)
    count_positive = np.sum(list_J_target > 0)
    if count_positive > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    elif tau <= 1:
        raise ValueError(
            "This protocol is not valid for fast protocol. "
            "Please choose tau > 1."
        )
    else:
        # Initialize result array
        density = np.zeros(len(list_h_target))
        
        # Compute ratio to determine which protocol to use for each
        ratio = np.divide(np.abs(list_h_target), np.abs(list_J_target))
        
        # Separate into ferromagnetic regime (ratio < 1)
        index_ferro = [i for i, v in enumerate(ratio) if v < 1]
        list_h_ferro = list_h_target[index_ferro]
        list_J_ferro = list_J_target[index_ferro]
        
        # Run ferromagnetic batch if any parameters in that regime
        if len(list_h_ferro) > 0:
            res = estimate_annealing_ferro_batch(
                nb_spins, list_h_ferro, list_J_ferro, periodic, tau, qpu
            )
            for i in range(len(index_ferro)):
                density[index_ferro[i]] = res[i]

        # Separate into paramagnetic regime (ratio >= 1)
        index_para = [i for i, v in enumerate(ratio) if v >= 1]
        list_h_para = list_h_target[index_para]
        list_J_para = list_J_target[index_para]

        # Run paramagnetic batch if any parameters in that regime
        if len(list_h_para) > 0:
            res = estimate_annealing_para_batch(
                nb_spins, list_h_para, list_J_para, periodic, tau, qpu
            )
            for i in range(len(index_para)):
                density[index_para[i]] = res[i]
        
    return density

def prepare_state_ferro(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Prepare a ferromagnetic state without measuring defects.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Target transverse field value
    J_target : float
        Ferromagnetic coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Annealing speed parameter (larger = slower)
    qpu : QPU
        Quantum processing unit (e.g., PyLinalg, LinAlg)
        
    Returns
    -------
    Result
        qat.core.Result object containing the final quantum state
    """
    # Build the two-phase schedule: rotation -> forward
    schedule_rot = sched_rot(-np.pi / 4, "Y", nb_spins)
    schedule_go = sched_go_ferro(nb_spins, h_target, J_target, periodic, tau)
    sched_fin = schedule_rot | schedule_go

    # Prepare initial ground state (symmetric or antisymmetric superposition)
    gs = np.zeros(2**nb_spins)
    if nb_spins % 2 == 0:
        gs[0] = 1 / np.sqrt(2)
        gs[-1] = 1 / np.sqrt(2)
    else:
        gs[0] = 1 / np.sqrt(2)
        gs[-1] = -1 / np.sqrt(2)

    # Submit job without observable (state preparation only)
    job_sample = sched_fin.to_job(psi_0=gs)
    res_sample = qpu.submit(job_sample)
    result = res_sample.join()
    
    return result


def prepare_state_para(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Prepare a paramagnetic state without measuring defects.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Transverse field value (constant during annealing)
    J_target : float
        Target coupling constant (should be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Annealing speed parameter (larger = slower)
    qpu : QPU
        Quantum processing unit (e.g., PyLinalg, LinAlg)
        
    Returns
    -------
    Result
        qat.core.Result object containing the final quantum state
    """
    # Build the single-phase schedule: forward only
    schedule_go = sched_go_para(nb_spins, h_target, J_target, periodic, tau)

    # Prepare initial state aligned with transverse field
    if h_target > 0:
        gs = '0' * nb_spins  # All spins up
    else:
        gs = '1' * nb_spins  # All spins down

    # Submit job without observable (state preparation only)
    job_sample = schedule_go.to_job(psi_0=gs)
    res_sample = qpu.submit(job_sample)
    result = res_sample.join()

    return result


def prepare_state(nb_spins, h_target, J_target, periodic, tau, qpu):
    """Prepare a state using the appropriate annealing protocol.
    
    Automatically selects ferromagnetic or paramagnetic protocol based on
    the relative magnitudes of h_target and J_target.
    
    Parameters
    ----------
    nb_spins : int
        Number of spins in the system
    h_target : float
        Transverse field target value
    J_target : float
        Coupling target value (must be negative)
    periodic : bool
        Whether to use periodic boundary conditions
    tau : float
        Annealing speed parameter (must be > 1)
    qpu : QPU
        Quantum processing unit (e.g., PyLinalg, LinAlg)
        
    Returns
    -------
    Result
        qat.core.Result object containing the final quantum state
        
    Raises
    ------
    ValueError
        If J_target > 0 (anti-ferromagnetic) or tau <= 1
    """
    # Validate input parameters
    if np.sign(J_target) > 0:
        raise ValueError(
            "This protocol is not valid for the Anti-Ferromagnetic "
            "interaction. Please choose J < 0."
        )
    elif tau <= 1:
        raise ValueError(
            "This protocol is not valid for fast protocol. "
            "Please choose tau > 1."
        )
    else:
        # Select protocol based on dominant parameter
        if np.abs(h_target) > np.abs(J_target):
            result = prepare_state_para(
                nb_spins, h_target, J_target, periodic, tau, qpu
            )
        else:
            result = prepare_state_ferro(
                nb_spins, h_target, J_target, periodic, tau, qpu
            )
    
    return result
