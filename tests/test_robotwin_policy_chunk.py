from __future__ import annotations

from collections import deque

import numpy as np

from robotwin_eval.policy_wrapper import HyVLAPolicyWrapper


def test_fresh_chunk_discards_legacy_cached_suffix():
    wrapper = HyVLAPolicyWrapper.__new__(HyVLAPolicyWrapper)
    wrapper.use_video_encoder = False
    wrapper.default_execute_steps = 3
    wrapper.action_cache = deque()
    calls = []

    def predict(_batch):
        calls.append(len(calls) + 1)
        base = 10 * calls[-1]
        return np.repeat(np.arange(base, base + 5)[:, None], 16, axis=1).astype(
            np.float32
        )

    wrapper._predict_fresh_chunk = predict
    batch = {"observation.state": np.zeros((1, 32), dtype=np.float32)}

    assert wrapper.get_action(batch)[0] == 10
    assert wrapper.get_action(batch)[0] == 11
    assert calls == [1]

    fresh = wrapper.get_action_chunk(batch, max_actions=2)
    np.testing.assert_array_equal(fresh[:, 0], [20, 21])
    assert not wrapper.action_cache

    # The legacy suffix value 12 must never appear after the fresh call.
    assert wrapper.get_action(batch)[0] == 30
    assert calls == [1, 2, 3]
