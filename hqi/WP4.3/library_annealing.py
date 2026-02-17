import numpy as np
import time
import sys
from qat.core import Observable, Term, Schedule
from qat.core.variables import Variable


def write_ising_ham(n, h, J, periodic):
    """
    Builds the operators, positions, and coefficients for an Ising Hamiltonian of the form: H = h * sum(Z_i) + J * sum(XX_{i,i+1})

    Parameters:
        n          int     Number of spins
        h          float   Transverse field coefficient (can be a Variable for scheduling)
        J          float   Coupling coefficient (can be a Variable for scheduling)
        periodic   bool    If True, adds a coupling between the last and first spin

    Returns:
        operator   list    Pauli operator strings (e.g. 'Z', 'XX')
        position   list    Qubit positions for each term
        coeff      list    Coefficients for each term"""
    
    operator = ['0' for i in range((2*n-1))]
    operator[:n] = ['Z' for i in range(n)]
    operator[n:] = ['XX' for i in range(n, 2*n-1)]

    position = [[0] for i in range(2*n-1)]
    position[:n] = [[i] for i in range(n)]
    position[n:] = [[i, i+1] for i in range(n-1)]

    coeff = [[0] for i in range(2*n-1)]
    coeff[:n] = [h for i in range(n)]
    coeff[n:] = [J for i in range(n, 2*n-1)]

    if periodic and n > 2:
        operator.append('XX')
        position.append([n-1, 0])
        coeff.append(J)

    return operator, position, coeff

##################################################################

def sched_rot(angle, axes, n):
    """
    Creates a rotation schedule applied uniformly to all n qubits. Used to prepare the initial state before annealing, starting from the ferromagnetic side.

    Parameters:
      angle      float   Rotation angle (e.g. -pi/4)
      axes       str     Pauli axis for rotation (e.g. 'Y')
      n          int     Number of qubits

    Returns:
      Schedule object with tmax=1

    """
    Rot = [Term(angle, axes, [i]) for i in range(n)]
    rot_ham = Observable(n, pauli_terms=Rot)
    drive_rot = [(1, rot_ham)]
    schedule_rot = Schedule(drive=drive_rot, tmax=1)
    return schedule_rot

############################################

def density_def_ferro(n, periodic):
    """Defines the ferromagnetic kink operator, counts the number of neighbouring spins not aligned: rho = (1/N) * sum(XX_{i,i+1}). It is used to measure how close the system is to the ferromagnetic ground state.

    Parameters:
      n          int     Number of spins
      periodic   bool    If True, includes the XX term between last and first spin

    Returns:
      Observable (qat.core)
    """
    operator = ['XX' for i in range(n-1)]
    position = [[i, i+1] for i in range(n-1)]

    if periodic and n > 2:
        operator.append('XX')
        position.append([n-1, 0])

    coeff = [1/len(operator) for i in range(len(operator))]
    Density = [Term(coeff[i], operator[i], position[i]) for i in range(len(operator))]
    dens = Observable(n, pauli_terms=Density)
    return dens


def density_def_para(n):
    """Defines the paramagnetic kink operator, counts the number of spins aligned in the opposite direction of the transverse magnetic field: rho = (1/N) * sum(Z_{i}). It is used to measure how close the system is to the paramagnetic ground state.

    Parameters:
      n          int     Number of spins

    Returns:
      Observable (qat.core)
    """
    operator = ['Z' for i in range(n)]
    position = [[i] for i in range(n)]

    coeff = [1/len(operator) for i in range(len(operator))]
    Density = [Term(coeff[i], operator[i], position[i]) for i in range(len(operator))]
    dens = Observable(n, pauli_terms=Density)
    return dens


def quantum_annealing_ferro(nb_spins, h_target, J_target, periodic, tau, qpu):
    """
    Runs a ferromagnetic quantum annealing protocol. The ramp back down is used to measure the defects generated in a known basis, so if we do not accumulate defects in the final state it means that we were close to the istantaneous ground state all along the time evolution.

    The schedule has 3 phases:
      1. Initial rotation by -pi/4 around Y axis
      2. Ramp up h from 0 to h_target over time |h_target| * tau
      3. Ramp h back down to 0 over time |h_target| * tau

    The initial state is the symmetric/antisymmetric superposition:
      gs[0] = 1/sqrt(2), gs[-1] = +/-1/sqrt(2) (sign depends on parity of n)

    Parameters:
      nb_spins   int     Number of spins
      h_target   float   Target transverse field value
      J_target   float   Ferromagnetic coupling (should be < 0)
      periodic   bool    Periodic boundary conditions
      tau        float   Annealing speed parameter
      qpu        QPU     Quantum processing unit to run the job

    Returns:
      float   Defect density = (1 - <rho_ferro>) / 2"""

    start = time.time()

    t = Variable("t", float)

    schedule_rot = sched_rot(-np.pi/4, "Y", nb_spins)

    h_go = np.sign(h_target)*t/tau
    time_go = np.abs(h_target)*tau
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(nb_spins, h_go, J_target, periodic)
    Hamiltonian = [Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i]) for i in range(len(oper_ham))]
    ham = Observable(nb_spins, pauli_terms=Hamiltonian)
    drive = [(1, ham)]
    schedule_go = Schedule(drive=drive, tmax=time_go)

    h_back = h_target-np.sign(h_target)*t/tau
    time_back = np.abs(h_target)*tau
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(nb_spins, h_back, J_target, periodic)
    Hamiltonian = [Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i]) for i in range(len(oper_ham))]
    ham = Observable(nb_spins, pauli_terms=Hamiltonian)
    drive = [(1, ham)]
    schedule_back = Schedule(drive=drive, tmax=time_back)

    sched_fin = schedule_rot | schedule_go | schedule_back

    gs = np.zeros(2**nb_spins)
    if nb_spins % 2 == 0:
        gs[0] = 1/np.sqrt(2)
        gs[-1] = 1/np.sqrt(2)
    else:
        gs[0] = 1/np.sqrt(2)
        gs[-1] = -1/np.sqrt(2)

    job_sample = sched_fin.to_job(psi_0=gs, observable=density_def_ferro(nb_spins, periodic))
    res_sample = qpu.submit(job_sample)
    job_num = res_sample.job_id
    res_sample.join()

    stop = time.time()
    # print('Run Time:%s'%(stop-start))
    return (1-res_sample.value)/2

    
def quantum_annealing_para(nb_spins, h_target, J_target, periodic, tau, qpu):
    """
    Runs a paramagnetic quantum annealing protocol. The ramp back down is used to measure the defects generated in a known basis, so if we do not accumulate defects in the final state it means that we were close to the istantaneous ground state all along the time evolution.

    The schedule has 2 phases:
      1. Ramp up |J| from 0 to J_target over time |J_target| * tau
      2. Ramp J back down to 0 over time |J_target| * tau

    Initial state:
      '000...0' if h_target > 0 (all spins up)
      '111...1' if h_target < 0 (all spins down)

    Parameters:
      nb_spins   int     Number of spins
      h_target   float   Transverse field value (constant throughout)
      J_target   float   Target coupling value (should be < 0)
      periodic   bool    Periodic boundary conditions
      tau        float   Annealing speed parameter
      qpu        QPU     Quantum processing unit to run the job

    Returns:
      float   Defect density:
                1 - <rho_para>  if h_target > 0
                1 + <rho_para>  if h_target < 0


    """
    start = time.time()

    t = Variable("t", float)

    J_go = -t/tau
    time_go = np.abs(J_target*tau)
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(nb_spins, h_target, J_go, periodic)
    Hamiltonian = [Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i]) for i in range(len(oper_ham))]
    ham = Observable(nb_spins, pauli_terms=Hamiltonian)
    drive = [(1, ham)]
    schedule_go = Schedule(drive=drive, tmax=time_go)

    J_back = J_target+t/tau
    time_back = np.abs(J_target*tau)
    oper_ham, pos_oper_ham, coeff_ham = write_ising_ham(nb_spins, h_target, J_back, periodic)
    Hamiltonian = [Term(coeff_ham[i], oper_ham[i], pos_oper_ham[i]) for i in range(len(oper_ham))]
    ham = Observable(nb_spins, pauli_terms=Hamiltonian)
    drive = [(1, ham)]
    schedule_back = Schedule(drive=drive, tmax=time_back)

    sched_fin = schedule_go | schedule_back
    if h_target > 0:
        psi_in = '0'*nb_spins
    else:
        psi_in = '1'*nb_spins

    job_sample = sched_fin.to_job(psi_0=psi_in, observable=density_def_para(nb_spins))
    res_sample = qpu.submit(job_sample)
    job_num = res_sample.job_id
    res_sample.join()

    if h_target > 0:
        result = 1-res_sample.value
    else:
        result = 1+res_sample.value


    stop = time.time()
    # print('Run Time:%s'%(stop-start))
    return result


def quantum_annealing_ising(nb_spins, h_target, J_target, periodic, tau, qpu):
    """
    Automatically selects the correct annealing protocol based on the relative magnitudes of h_target and J_target. Bear in mind that since we are also going back to the initial state if we choose a tau too small we would get very few defects (The state does not change), still it is not a reliable number. Tau should be at least of the order of unity.

    |h| > |J|  -->  paramagnetic protocol (quantum_annealing_para)
    |h| < |J|  -->  ferromagnetic protocol (quantum_annealing_ferro)

    Note: Anti-ferromagnetic coupling (J > 0) is not supported.
          The function will print an error and exit if J_target > 0.

    Parameters:
      nb_spins   int     Number of spins
      h_target   float   Transverse field target value
      J_target   float   Coupling target value (must be < 0)
      periodic   bool    Periodic boundary conditions
      tau        float   Annealing speed parameter (larger = slower = fewer defects)
      qpu        QPU     Quantum processing unit to run the job

    Returns:
      float   Defect density (between 0 and 1)

    """

    if np.sign(J_target) > 0:
        print("This protocol is not valid for the Anti-Ferromagnetic interaction.\n Please choose J<0.")
        sys.exit()
    elif tau < 1:
        print("This protocol is not valid for fast protocol.\n Please choose tau>1.")
        sys.exit()
    else:
        if np.abs(h_target) > np.abs(J_target):
            defects = quantum_annealing_para(nb_spins, h_target, J_target, periodic, tau, qpu)
        else:
            defects = quantum_annealing_ferro(nb_spins, h_target, J_target, periodic, tau, qpu)

    return defects
    
