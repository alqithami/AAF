"""These tests require the ACTUAL engine. No fake simulator fallback."""
import importlib.util,os
import pytest
import torch
from aaf_strengthen.preflight import real_engine_check
pytestmark=pytest.mark.skipif(importlib.util.find_spec('vmas') is None,reason='Actual VMAS is not installed locally')

def test_actual_vmas_free_motion_and_filter():
    assert real_engine_check(torch.device(os.environ.get('AAF_TEST_DEVICE','cpu')))['status']=='PASS'

def test_actual_vmas_tiny_training_and_trace(tmp_path):
    from aaf_strengthen.protocol import plan
    from aaf_strengthen.navigation import run_navigation
    from aaf_strengthen.analysis import audit_arrays
    cfg=plan('smoke')[1]
    result,arrays=run_navigation(cfg,torch.device(os.environ.get('AAF_TEST_DEVICE','cpu')),tmp_path)
    assert len(result['episodes'])==2
    assert len(result['policy']['training_trace'])==2
    assert audit_arrays(arrays,cfg)['authority']=='PASS'
