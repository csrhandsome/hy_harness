from __future__ import annotations

import pytest
from robots.robotwin.prompt import PRIMITIVE_FIRST, VLA_FIRST, system_prompt


def test_prompt_profiles_have_shared_chunk_contract_and_distinct_policy():
    primitive = system_prompt(PRIMITIVE_FIRST)
    vla = system_prompt(VLA_FIRST)
    assert primitive["CHUNK_SEMANTICS"] == vla["CHUNK_SEMANTICS"]
    assert "Primitive-first" in primitive["CONTROL_POLICY"]
    assert "VLA-first" in vla["CONTROL_POLICY"]
    assert "Alternate strictly" not in " ".join(primitive.values())
    termination = primitive["SAFETY_AND_TERMINATION"]
    assert "EVERY assistant response MUST contain" in termination
    assert "failure, stuck, failed, incomplete" in termination
    assert "Only after robotwin_status reports success=true" in termination
    assert "Do not write an audit" in primitive["ARTIFACTS"]


def test_unknown_prompt_profile_is_rejected():
    with pytest.raises(ValueError):
        system_prompt("unknown")
