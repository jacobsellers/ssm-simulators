"""Gated RDM model configuration"""

# from ssms.basic_simulators.gated_rdm import gated_rdm_simulator, gated_rdm_stimcode_simulator

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_gated_rdm_config():
    """Get configuration for Gated RDM model."""
    return {
        "name": "gated_rdm",
        "params": ["vtask", "vsig", "vcom", "a", "ztask", "zresp", "trialtypecode", "kgate", "t"],
        "param_bounds": [
            [0.1, 0.1, 0.1, 0.5, 0.0, 0.0, -1, 0.1, 0.1],  # Lower bounds
            [8.0, 8.0, 5.0, 5.0, 3.0, 0.9, 1, 5.0, 1.5],  # Upper bounds
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 9,
        "default_params": [5.0, 2.0, 0.3, 1.5, 1.5, 0.2, 0, 1.0, 0.3],
        "nchoices": 4,  # Single-choice model
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.gated_racing_diffusion_model_stimcode,  # cython
        # "simulator": gated_rdm_stimcode_simulator,  # python
        # Define any parameter transforms here
        "parameter_transforms": {
            "sampling": [],  # No sampling constraints
            "simulation": [],  # No simulation transforms needed
        },
    }
