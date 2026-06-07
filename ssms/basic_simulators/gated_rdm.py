import numpy as np

DTYPE = np.float32


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def gated_rdm_simulator(
    vtask,  # task drift rate
    vsig,  # signaled drift rate (stim dimension)
    vcom,  # competition drift rate (wrong stim dimension)
    a,  # boundary
    x0task0,  # starting point for task accumulator 0
    x0task1,  # starting point for task accumulator 1
    # ztask,  # task bias
    # isrep,  # 1 if repeat, -1 if switch, 0 if first trial
    x0resp01,  # starting point for motor accumulator 0 and 1 (task 0)
    x0resp23,  # starting point for motor accumulator 2 and 3 (task 1)
    kgate,  # task gate
    toffset,  # task accumulator onset offset relative to response onset
    t,  # non decision time
    # s,  # decision noise
    delta_t=0.001,
    max_t=20.0,
    n_samples=1000,
    n_trials=1,
    random_state=None,
    return_option="full",
    **kwargs,
):

    # number of decision processes. 2 task accumulators + 4 motor accumulators
    n_particles = 6

    #
    delta_t_sqrt = np.sqrt(delta_t)
    # sqrt_st = delta_t_sqrt * s
    sqrt_st = delta_t_sqrt * 1

    # Allocate output arrays
    total_samples = n_trials * n_samples
    rts = np.zeros((n_samples, n_trials, 1), dtype=np.float32)
    choices = np.ones((n_samples, n_trials, 1), dtype=np.int8)

    # Trajectory storage - disabled in parallel mode
    traj = np.zeros((int(max_t / delta_t) + 1, n_particles), dtype=DTYPE)
    traj[:, :] = -999

    # convert task/motor accumulators to array
    drift_zeros = np.zeros(n_trials)

    v_task_arr = np.column_stack([vtask, drift_zeros])

    v_mot_task1 = np.column_stack([vsig, vcom, drift_zeros, drift_zeros])
    v_mot_task2 = np.column_stack([drift_zeros, drift_zeros, vsig, vcom])

    # init task accumulators
    x_task_init = np.column_stack([x0task0, x0task1])

    # init rng
    rng = np.random.default_rng(random_state)

    for flat_idx in range(total_samples):
        k = flat_idx // n_samples  # trial index
        n = flat_idx % n_samples  # sample index

        if flat_idx % 1000 == 0:
            print(f"Simulating sample {n} of trial {k} (flat index {flat_idx})")

        t_particle = 0.0
        ix = 0
        winner = -1
        winner_found = 0

        x_task_t = x_task_init[k, :].copy()
        x_motor_t = np.array([x0resp01[k], x0resp01[k], x0resp23[k], x0resp23[k]])

        task_pre_time = 0.0
        while task_pre_time < toffset[k] and task_pre_time < max_t:
            x_task_t += v_task_arr[k, :] * delta_t + sqrt_st * rng.standard_normal(2)
            task_pre_time += delta_t

        task_start_time = max(0.0, -toffset[k])

        # Race simulation (first-past-the-post, no reflecting boundary)
        while not winner_found and t_particle <= max_t:
            if t_particle >= task_start_time:
                x_task_t += v_task_arr[k, :] * delta_t + sqrt_st * rng.standard_normal(2)

            w0 = logistic(kgate[k] * (x_task_t[0] - x_task_t[1]))
            w1 = 1 - w0

            x_motor_t += (
                w0 * v_mot_task1[k, :] * delta_t + w1 * v_mot_task2[k, :] * delta_t + sqrt_st * rng.standard_normal(4)
            )

            t_particle += delta_t
            ix += 1

            if np.any(x_motor_t >= a[k]):
                winner = np.where(x_motor_t >= a[k])[0][0]
                winner_found = 1
                break

            if winner_found:
                break

            if t_particle >= max_t:
                break

        rts[n, k, 0] = t_particle + t[k]
        choices[n, k, 0] = winner
        if rts[n, k, 0] == -999 or (not winner_found):
            rts[n, k, 0] = -999
            choices[n, k, 0] = -1

    if return_option == "full":
        metadata = {
            "model": "gated_rdm",
            "params": [
                "vtask",
                "vsig",
                "vcom",
                "a",
                "x0task0",
                "x0task1",
                "x0resp01",
                "x0resp23",
                "kgate",
                "toffset",
                "t",
            ],
            "param_values": {
                "vtask": vtask,
                "vsig": vsig,
                "vcom": vcom,
                "a": a,
                "x0task0": x0task0,
                "x0task1": x0task1,
                "x0resp01": x0resp01,
                "x0resp23": x0resp23,
                "kgate": kgate,
                "toffset": toffset,
                "t": t,
            },
            "n_samples": n_samples,
            "n_trials": n_trials,
            "max_t": max_t,
            "possible_choices": [0, 1, 2, 3],
        }
    else:
        metadata = {
            "model": "gated_rdm",
            "max_t": max_t,
            "possible_choices": [0, 1, 2, 3],
        }
    return {
        "rts": rts,
        "choices": choices,
        "metadata": metadata,
    }


def gated_rdm_stimcode_simulator(
    vtask,  # task drift rate
    vsig,  # signaled drift rate (stim dimension)
    vcom,  # competition drift rate (wrong stim dimension)
    a,  # boundary
    ztask,  # task bias
    trialtypecode,  # 1 if repeat, -1 if switch, 0 if first trial
    kgate,  # task gate
    zresp,
    toffset,  # task accumulator onset offset relative to response onset
    t,  # non decision time
    delta_t=0.001,
    max_t=20.0,
    n_samples=1000,
    n_trials=1,
    random_state=None,
    return_option="full",
    **kwargs,
):
    """
    theta columns:
      0  vtask
      1  vsig
      2  vcom
      3  a
      4  ztask
      5  trialtypecode   (-1 switch, 0 first/neither, +1 repeat)
      6  kgate
      7  zresp
      8  toffset
      9  t
    """

    valid = np.isin(trialtypecode, [-1, 0, 1])
    if not np.all(valid):
        bad = np.unique(trialtypecode[~valid])
        raise ValueError(f"Invalid trialtypecode values: {bad}")

    x0task0 = ztask * (trialtypecode == 1)
    x0task1 = ztask * (trialtypecode == -1)

    x0resp01 = zresp * (trialtypecode == 1)
    x0resp23 = zresp * (trialtypecode == -1)

    return gated_rdm_simulator(
        vtask=vtask,
        vsig=vsig,
        vcom=vcom,
        a=a,
        x0task0=x0task0.astype(np.float32),
        x0task1=x0task1.astype(np.float32),
        x0resp01=x0resp01.astype(np.float32),
        x0resp23=x0resp23.astype(np.float32),
        kgate=kgate,
        toffset=toffset,
        t=t,
        delta_t=delta_t,
        max_t=max_t,
        n_samples=n_samples,
        n_trials=n_trials,
        random_state=random_state,
        return_option=return_option,
        **kwargs,
    )
