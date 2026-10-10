import ctypes

import pytest
from types import SimpleNamespace

import llama_cpp
import llama_cpp.llama_cpp as llama_cpp_lib


@pytest.mark.parametrize("value", [0, 256, 1024])
def test_parse_fit_target_mib(value):
    assert llama_cpp.Llama._parse_fit_target_mib(value) == value


@pytest.mark.parametrize("value", [-1, True, 1.5, "256"])
def test_parse_fit_target_mib_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        llama_cpp.Llama._parse_fit_target_mib(value)


@pytest.mark.parametrize("value", [0, 1024, 4096])
def test_parse_fit_min_ctx(value):
    assert llama_cpp.Llama._parse_fit_min_ctx(value) == value


@pytest.mark.parametrize("value", [-1, True, 1.5, "2048"])
def test_parse_fit_min_ctx_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        llama_cpp.Llama._parse_fit_min_ctx(value)


def test_fit_model_params_passes_target_margin(monkeypatch):
    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = 0
    cparams = llama_cpp.llama_context_default_params()
    calls = []

    def fit_params(path, model_params, context_params, tensor_split, overrides, margins, min_ctx):
        calls.append((path, model_params, context_params, list(margins), min_ctx))
        return 0

    monkeypatch.setattr(llama_cpp_lib, "llama_common_fit_params", fit_params)
    instance = SimpleNamespace(
        fit=True,
        model_params=params,
        context_params=cparams,
        model_path="model.gguf",
        fit_target_mib=256,
        fit_min_ctx=2048,
        _fit_tensor_split=None,
        _fit_tensor_buft_overrides=None,
        _fit_margins=None,
    )

    llama_cpp.Llama._fit_model_params(instance)

    assert len(calls) == 1
    assert calls[0][0] == b"model.gguf"
    assert calls[0][1] is params
    assert calls[0][2] is cparams
    assert calls[0][3] == [256 * 1024 * 1024] * llama_cpp_lib.LLAMA_MAX_DEVICES
    assert calls[0][4] == 2048
    assert instance._fit_tensor_split is not None
    assert instance._fit_tensor_buft_overrides is not None


def test_fit_model_params_requires_auto():
    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = 1
    instance = SimpleNamespace(fit=True, model_params=params)

    with pytest.raises(ValueError, match="n_gpu_layers=0"):
        llama_cpp.Llama._fit_model_params(instance)


def test_fit_model_params_surfaces_failure(monkeypatch):
    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = 0
    monkeypatch.setattr(llama_cpp_lib, "llama_common_fit_params", lambda *args: 1)
    instance = SimpleNamespace(
        fit=True,
        model_params=params,
        context_params=llama_cpp.llama_context_default_params(),
        model_path="model.gguf",
        fit_target_mib=1024,
        fit_min_ctx=4096,
        _fit_tensor_split=None,
        _fit_tensor_buft_overrides=None,
        _fit_margins=None,
    )

    with pytest.raises(RuntimeError, match="status 1"):
        llama_cpp.Llama._fit_model_params(instance)
