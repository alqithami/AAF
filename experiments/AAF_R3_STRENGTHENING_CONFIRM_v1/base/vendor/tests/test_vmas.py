import importlib.util
import pytest,torch
from aaf_r3.physics import preflight_navigation
@pytest.mark.skipif(importlib.util.find_spec("vmas") is None,reason="VMAS not installed: actual-engine check must run on target machine")
def test_actual_vmas_api():
    assert preflight_navigation(torch.device("cpu"))["status"]=="PASS"


@pytest.mark.skipif(importlib.util.find_spec("vmas") is None,reason="VMAS not installed: actual-engine training/evaluation check must run on target machine")
def test_real_navigation_training_and_evaluation(tmp_path):
    import torch
    from aaf_r3.cli import plan
    from aaf_r3.physics import run_navigation
    torch.set_num_threads(1)
    cfg=plan("smoke","physics")[0]
    result=run_navigation(cfg,torch.device("cpu"),tmp_path)
    assert len(result["episodes"])==2
    assert result["training_trace"]
    assert 0 <= result["summary"]["collision_episode"] <= 1
    assert 0 <= result["summary"]["attack_exposed"] <= 1
    assert result["provenance"]["packages"]["vmas"]=="1.5.2"
