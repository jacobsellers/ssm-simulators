# Global settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Gated Race Models

This module contains simulator functions for gated race models, where multiple
accumulators race independently toward their own decision boundaries.
Unlike DDM, gated race models have separate evidence accumulators for each choice.

The gating mechanism allows the model to flexibly weight evidence for different choices based on the relative activation of task-relevant accumulators. This can capture
dynamic shifts in attention
"""

import cython
from libc.math cimport sqrt, log, fmax, exp
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np

# OpenMP imports
#from cython.parallel cimport prange, parallel, threadid
#from cssm._openmp_status import check_parallel_request

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    draw_uniform,
    draw_gaussian,
    random_uniform,
    sign,
    csum,
    compute_boundary,
    compute_smooth_unif,
    enforce_deadline,
    compute_deadline_tmp,
    build_param_dict_from_2d_array,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)

DTYPE = np.float32




# Race Model ------------------------------------

def gated_racing_diffusion_model(
    np.ndarray[float, ndim = 1] vtask,
    np.ndarray[float, ndim = 1] vsig,
    np.ndarray[float, ndim = 1] vcom,
    np.ndarray[float, ndim = 1] a,
    np.ndarray[float, ndim = 1] x0task0,
    np.ndarray[float, ndim = 1] x0task1,
    np.ndarray[float, ndim = 1] x0resp01,
    np.ndarray[float, ndim = 1] x0resp23,
    np.ndarray[float, ndim = 1] kgate,
    np.ndarray[float, ndim = 1] t,
    #np.ndarray[float, ndim = 1] deadline,
    float delta_t = 0.001,
    float max_t = 20,
    int n_samples = 2000,
    int n_trials = 1,
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
    int n_threads = 1,
    **kwargs
):
    set_seed(random_state)

    # param views

    cdef float[:] vtask_view = vtask
    cdef float[:] vsig_view = vsig
    cdef float[:] vcom_view = vcom
    cdef float[:] a_view = a
    cdef float[:] x0task0_view = x0task0
    cdef float[:] x0task1_view = x0task1
    cdef float[:] x0resp01_view = x0resp01
    cdef float[:] x0resp23_view = x0resp23
    cdef float[:] kgate_view = kgate
    cdef float[:] t_view = t
    # cdef float[:] deadline_view = deadline

    cdef float sqrt_st = sqrt(delta_t)
    #sqrt_st = delta_t_sqrt * s

    cdef int n_particles = 6 # two task + 4 motor accumulators
    rts = np.zeros((n_samples, n_trials, 1), dtype = DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype = np.intc)
    cdef int[:, :, :] choices_view = choices

    particles = np.zeros((n_particles), dtype = DTYPE)
    cdef float[:] particles_view = particles
    wtask = np.zeros(2, dtype=DTYPE)
    cdef float[:] wtask_view = wtask

    # Trajectory saving (for first trial, first sample)
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype = DTYPE)
    traj[:, :] = -999
    cdef float[:, :] traj_view = traj

    # Initialize variables needed for for loop
    cdef float t_particle, smooth_u # , deadline_tmp
    cdef Py_ssize_t n, ix, j, k
    cdef Py_ssize_t m = 0
    cdef int winner = -1
    cdef int winner_found = 0

    cdef int num_steps = int((max_t / delta_t) + 1)
    cdef int num_draws = num_steps * n_particles
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    # cdef Py_ssize_t mu = 0
    # cdef float[:] uniform_values = draw_uniform(num_draws)

    for k in range(n_trials):

        #deadline_tmp = compute_deadline_tmp(max_t, deadline_view[k], t_view[k])

        # Loop over samples
        for n in range(n_samples):

            t_particle = 0.0
            ix = 0
            winner = -1
            winner_found = 0

            particles_view[0] = x0task0_view[k]
            particles_view[1] = x0task1_view[k]
            #wtask_view[0] = 1 / (1 + exp(-kgate_view[k] * (particles_view[0] - particles_view[1])))
            #wtask_view[1] = 1 - wtask_view[0]
            particles_view[2] = x0resp01_view[k]
            particles_view[3] = x0resp01_view[k]
            particles_view[4] = x0resp23_view[k]
            particles_view[5] = x0resp23_view[k]

            while not winner_found and t_particle <= max_t:

                particles_view[0] += (vtask_view[k] * delta_t) + sqrt_st * gaussian_values[m]
                particles_view[1] += sqrt_st * gaussian_values[m+1]

                wtask_view[0] = 1 / (1 + exp(-kgate_view[k] * (particles_view[0] - particles_view[1])))
                wtask_view[1] = 1 - wtask_view[0]

                particles_view[2] += (vsig_view[k] * wtask_view[0] * delta_t) + sqrt_st * gaussian_values[m+2]
                particles_view[3] += (vcom_view[k] * wtask_view[0] * delta_t) + sqrt_st * gaussian_values[m+3]
                particles_view[4] += (vsig_view[k] * wtask_view[1] * delta_t) + sqrt_st * gaussian_values[m+4]
                particles_view[5] += (vcom_view[k] * wtask_view[1] * delta_t) + sqrt_st * gaussian_values[m+5]

                m += 6
                if m >= num_draws:
                    m = 0
                    gaussian_values = draw_gaussian(num_draws)

                # check for winner
                for j in range(2, 6):
                    if particles_view[j] >= a_view[k]:
                        winner_found = 1
                        winner = j - 2
                        break
                
                if winner_found:
                    break

                t_particle += delta_t
                ix += 1

            # Store RT and choice
            rts_view[n, k, 0] = t_particle + t[k]
            choices_view[n, k, 0] = winner

            # Handle non-responses (deadline hit or no decision)
            # enforce_deadline(rts_view, deadline_view, n, k, 0)
            if rts_view[n, k, 0] == -999 or (not winner_found):
                rts_view[n, k, 0] = -999
                choices_view[n, k, 0] = -1

        
    # Build minimal metadata first
    minimal_meta = build_minimal_metadata(
        simulator_name='gated_rdm_simulator',
        possible_choices=[0,1,2,3],
        n_samples=n_samples,
        n_trials=n_trials,
    )
    if return_option == 'full':

        sim_config = {'delta_t': delta_t, 'max_t': max_t, 'n_threads': 1}
        params = {
                "vtask": vtask,
                "vsig": vsig,
                "vcom": vcom,
                "a": a,
                "x0task0": x0task0,
                "x0task1": x0task1,
                "x0resp01": x0resp01,
                "x0resp23": x0resp23,
                "kgate": kgate,
                "t": t,
            }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            traj=traj,
            extra_params=None
        )
        return build_return_dict(rts, choices, full_meta)

    elif return_option == 'minimal':
        return build_return_dict(rts, choices, minimal_meta)

    else:
        raise ValueError('return_option must be either "full" or "minimal"')



def gated_racing_diffusion_model_stimcode(
    np.ndarray[float, ndim = 1] vtask,
    np.ndarray[float, ndim = 1] vsig,
    np.ndarray[float, ndim = 1] vcom,
    np.ndarray[float, ndim = 1] a,
    np.ndarray[float, ndim = 1] ztask,
    np.ndarray[float, ndim = 1] zresp,
    np.ndarray[float, ndim = 1] trialtypecode,
    np.ndarray[float, ndim = 1] kgate,
    np.ndarray[float, ndim = 1] t,
    # np.ndarray[float, ndim = 1] deadline,
    float delta_t = 0.001,
    float max_t = 20,
    int n_samples = 2000,
    int n_trials = 1,
    random_state = None,
    return_option = 'full',
    smooth_unif = False,
    int n_threads = 1,
    **kwargs
):

    valid = np.isin(trialtypecode, [-1, 0, 1])
    if not np.all(valid):
        bad = np.unique(trialtypecode[~valid])
        raise ValueError(f"Invalid trialtypecode values found: {bad}. Allowed values are -1, 0, 1.")

    x0task0 = ztask * (trialtypecode == 1)
    x0task1 = ztask * (trialtypecode == -1)

    x0resp01 = zresp * a * (trialtypecode == 1)
    x0resp23 = zresp * a * (trialtypecode == -1)

    return gated_racing_diffusion_model(
        vtask=vtask,
        vsig=vsig,
        vcom=vcom,
        a=a,
        x0task0=x0task0,
        x0task1=x0task1,
        x0resp01=x0resp01,
        x0resp23=x0resp23,
        kgate=kgate,
        t=t,
        # deadline=deadline,
        delta_t=delta_t,
        max_t=max_t,
        n_samples=n_samples,
        n_trials=n_trials,
        random_state=random_state,
        return_option=return_option,
        smooth_unif=smooth_unif,
        n_threads=n_threads
    )

# -----------------------------------------------------------------------------------------------