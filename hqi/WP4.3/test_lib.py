"""Test suite for quantum annealing and measurement libraries."""

import numpy as np
import pytest
from qlmaas.qpus import AnalogQPU
from qat.core import Observable

import library_annealing as ann
import library_measurments as mes


def test_estimate_state_para():
    """Test paramagnetic state quality estimation with zero defects."""
    nb_spins = 4
    h_target = 10
    J_target = 0
    tau = 1.3
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))
    
    result = ann.estimate_state_quality(
        nb_spins, h_target, J_target, periodic, tau, qpu
    )
    
    assert np.allclose(result, 0, atol=1e-6)


def test_estimate_state_ferro():
    """Test ferromagnetic state quality estimation with zero defects."""
    nb_spins = 4
    h_target = 0
    J_target = -1
    tau = 1.3
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))
    
    result = ann.estimate_state_quality(
        nb_spins, h_target, J_target, periodic, tau, qpu
    )
    
    assert np.allclose(result, 0, atol=1e-6)


def test_measure_para_one_point():
    """Test one-point correlation in paramagnetic regime."""
    nb_spins = 4
    h_target = 10
    J_target = 0
    tau = 1.3
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))

    direction = 'Z'
    list_position = [i for i in range(nb_spins)]

    schedule_go = ann.sched_go_para(
        nb_spins, h_target, J_target, periodic, tau
    )
    gs = '0' * nb_spins
    
    result = mes.one_point_corr(
        schedule_go, direction, list_position, qpu, gs
    )
    
    assert np.allclose(result, 1, atol=1e-6)


def test_measure_ferro_two_point():
    """Test two-point correlation in ferromagnetic regime."""
    nb_spins = 4
    h_target = 0
    J_target = -1
    tau = 1.3
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))

    direction = 'X'
    list_position = [[i, i + 1] for i in range(nb_spins - 1)]
    
    schedule_rot = ann.sched_rot(-np.pi / 4, 'Y', nb_spins)
    schedule_go = ann.sched_go_ferro(
        nb_spins, h_target, J_target, periodic, tau
    )
    schedule_final = schedule_rot | schedule_go
    
    gs = np.zeros(2**nb_spins)
    gs[0] = 1 / np.sqrt(2)
    gs[-1] = 1 / np.sqrt(2)
    
    result = mes.two_point_corr(
        schedule_final, direction, list_position, qpu, gs
    )
    
    assert np.allclose(result, 1, atol=1e-6)


def test_random_measure_one_point():
    """Test one-point correlation against Observable method with random state."""
    nb_spins = 4
    h_target = np.random.uniform(-3, -1.1)
    J_target = np.random.uniform(-1, 0)
    tau = np.random.uniform(1.1, 10)
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))

    direction = 'Z'
    list_position = [i for i in range(nb_spins)]

    schedule_go = ann.sched_go_para(
        nb_spins, h_target, J_target, periodic, tau
    )
    
    state = (
        np.random.uniform(-1, 1, 2**nb_spins) +
        1.j * np.random.uniform(-1, 1, 2**nb_spins)
    )
    state_norm = state / np.linalg.norm(state)
    
    val = mes.one_point_corr(
        schedule_go, direction, list_position, qpu, state_norm
    )

    list_obs = [Observable.z(i, nb_spins) for i in list_position]
    job_obs = schedule_go.to_job(
        psi_0=state_norm, observable=list_obs[0], observables=list_obs[1:]
    )
    result = qpu.submit(job_obs)
    val_compare = [result.value] + result.values
    
    assert np.allclose(val, val_compare, atol=1e-6)


def test_random_measure_two_point():
    """Test two-point correlation against Observable method with random state."""
    nb_spins = 4
    h_target = np.random.uniform(-3, -1.1)
    J_target = np.random.uniform(-1, 0)
    tau = np.random.uniform(1.1, 10)
    periodic = True
    qpu = AnalogQPU(error_control=True, abs_tol=10**(-8), rel_tol=10**(-8))

    direction = 'X'
    list_position = [[i, i + 1] for i in range(nb_spins - 1)]

    schedule_go = ann.sched_go_para(
        nb_spins, h_target, J_target, periodic, tau
    )
    
    state = (
        np.random.uniform(-1, 1, 2**nb_spins) +
        1.j * np.random.uniform(-1, 1, 2**nb_spins)
    )
    state_norm = state / np.linalg.norm(state)
    
    val = mes.two_point_corr(
        schedule_go, direction, list_position, qpu, state_norm
    )

    list_obs = [
        Observable.x(sublist[0], nb_spins) * Observable.x(sublist[1], nb_spins)
        for sublist in list_position
    ]
    job_obs = schedule_go.to_job(
        psi_0=state_norm, observable=list_obs[0], observables=list_obs[1:]
    )
    result = qpu.submit(job_obs)
    val_compare = [result.value] + result.values
    
    assert np.allclose(val, val_compare, atol=1e-6)
