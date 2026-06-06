import inspect

import numpy as np

import cssm


def _stimcode_params(toffset):
    return {
        "vtask": np.array([5.0], dtype=np.float32),
        "vsig": np.array([2.0], dtype=np.float32),
        "vcom": np.array([0.3], dtype=np.float32),
        "a": np.array([1.5], dtype=np.float32),
        "ztask": np.array([1.5], dtype=np.float32),
        "zresp": np.array([0.2], dtype=np.float32),
        "trialtypecode": np.array([0.0], dtype=np.float32),
        "kgate": np.array([1.0], dtype=np.float32),
        "toffset": np.array([toffset], dtype=np.float32),
        "t": np.array([0.3], dtype=np.float32),
    }


def test_gated_rdm_stimcode_accepts_toffset():
    signature = inspect.signature(cssm.gated_racing_diffusion_model_stimcode)

    assert "toffset" in signature.parameters
    assert list(signature.parameters).index("toffset") < list(
        signature.parameters
    ).index("t")


def test_gated_rdm_toffset_smoke():
    for toffset in [0.0, 0.05, -0.05]:
        out = cssm.gated_racing_diffusion_model_stimcode(
            **_stimcode_params(toffset),
            n_samples=5,
            n_trials=1,
            delta_t=0.01,
            max_t=1.0,
            random_state=123,
            return_option="full",
        )

        assert out["rts"].shape == (5, 1, 1)
        assert out["choices"].shape == (5, 1, 1)
        assert out["metadata"]["toffset"][0] == np.float32(toffset)
