# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import importlib
import inspect
import logging
import os
import shutil
from functools import partial
from typing import Any, Dict, List, Optional

import ray
from torch import nn
from transformers import AutoTokenizer

logger = logging.getLogger("ray")

# Device visibility variables used by CUDA and ROCm/DTK (Hygon DCU) runtimes.
_VISIBLE_DEVICE_ENV_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

_WEIGHT_FILE_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors")


# def get_directory_size(directory):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             if not os.path.islink(file_path):
#                 total_size += os.path.getsize(file_path)
#     return total_size


@ray.remote(num_cpus=1)
def download_transformers_model(
    model_name: str,
    pretrained_model_name_or_path: str,
    torch_dtype: str,
    hf_model_class: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    model_path = os.path.join(storage_path, "transformers", model_name)
    tokenizer_path = os.path.join(
        storage_path, "transformers", model_name, "tokenizer"
    )

    if os.path.exists(model_path):
        logger.info(f"{model_path} already exists")
        return True

    import torch

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    logger.info(f"Downloading {model_path}")

    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    model = hf_model_cls.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from sllm_store.transformers import save_model

    logger.info(f"Saving {model_path}")
    try:
        save_model(model, model_path)
        tokenizer.save_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to save {model_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {model_name} for transformer backend: {e}"
        )

    # model_size = get_directory_size(model_path)
    # logger.info(f"{model_name} (size: {model_size}) downloaded")

    return True


@ray.remote(num_cpus=1)
def download_lora_adapter(
    base_model_name: str,
    adapter_name: str,
    adapter_name_or_path: str,
    hf_model_class: str,
    torch_dtype: str,
) -> bool:
    storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    adapter_path = os.path.join(
        storage_path, "transformers", adapter_name_or_path
    )

    if os.path.exists(adapter_path):
        logger.info(f"{adapter_path} already exists")
        return True

    import torch

    torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        os.path.join(storage_path, "transformers", base_model_name),
        trust_remote_code=True,
    )
    config.torch_dtype = torch_dtype
    module = importlib.import_module("transformers")
    hf_model_cls = getattr(module, hf_model_class)
    base_model = hf_model_cls.from_config(
        config,
        trust_remote_code=True,
    ).to(config.torch_dtype)

    logger.info(f"Downloading {adapter_path}")
    from peft import PeftModel

    try:
        model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise RuntimeError(f"LoRA adapter load failed: {e}")

    from sllm_store.transformers import save_lora

    logger.info(f"Saving {adapter_path}")
    try:
        save_lora(model, adapter_path)
    except Exception as e:
        logger.error(f"Failed to save {adapter_path}: {e}")
        # shutil.rmtree(model_path)  # TODO: deal with error in save_model
        raise RuntimeError(
            f"Failed to save {adapter_name} for transformer backend: {e}"
        )

    return True


def _get_ray_gpu_ids() -> List[str]:
    """Return the GPU/DCU IDs allocated to the current Ray actor."""
    try:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        return [str(device_id) for device_id in accelerator_ids.get("GPU", [])]
    except Exception as exc:
        logger.warning("Unable to read Ray accelerator IDs: %s", exc)
        return []


def _get_visible_device_env() -> Dict[str, str]:
    return {
        env_name: os.environ[env_name]
        for env_name in _VISIBLE_DEVICE_ENV_VARS
        if env_name in os.environ
    }


def _is_rocm_accelerator(accelerator: Optional[str], torch_module: Any) -> bool:
    """Whether this downloader runs on a ROCm/HIP runtime (Hygon DCU, AMD)."""
    if accelerator in {"hygon", "dcu", "rocm"}:
        return True
    return getattr(getattr(torch_module, "version", None), "hip", None) is not None


def _sync_rocm_visible_devices(assigned_gpu_ids: List[str]) -> None:
    """Pin the Ray allocation into ROCR_VISIBLE_DEVICES (highest precedence
    in ROCm/DTK, devices re-index from 0).  Must run before torch device
    init; precedence verified on DTK 26.04 only.
    """
    if not assigned_gpu_ids:
        return

    physical_devices = ",".join(
        str(device_id) for device_id in assigned_gpu_ids
    )
    existing_rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
    if existing_rocr is not None:
        if sorted(existing_rocr.split(",")) != sorted(assigned_gpu_ids):
            raise RuntimeError(
                f"ROCR_VISIBLE_DEVICES={existing_rocr} conflicts with the "
                f"Ray allocation {physical_devices}; refusing to guess the "
                "device mapping"
            )
        return

    if _get_visible_device_env():
        # Visibility already pinned by Ray/deployment; trust it.
        logger.info(
            "Device visibility already configured (%s); left untouched",
            _get_visible_device_env(),
        )
        return

    os.environ["ROCR_VISIBLE_DEVICES"] = physical_devices
    logger.info(
        "Pinned ROCR_VISIBLE_DEVICES=%s from Ray allocation",
        physical_devices,
    )


def _check_rank_layout(model_path: str, tensor_parallel_size: int) -> None:
    """Structural check: all expected TP rank directories are present."""
    rank_directories = sorted(
        (
            entry
            for entry in os.listdir(model_path)
            if entry.startswith("rank_")
            and entry.removeprefix("rank_").isdigit()
            and os.path.isdir(os.path.join(model_path, entry))
        ),
        key=lambda entry: int(entry.removeprefix("rank_")),
    )
    expected_ranks = [
        f"rank_{rank}" for rank in range(tensor_parallel_size)
    ]
    if rank_directories != expected_ranks:
        raise RuntimeError(
            f"ServerlessLLM checkpoint at {model_path} is incomplete or "
            "uses a different tensor parallel size: expected="
            f"{expected_ranks}, actual={rank_directories}. If this is a "
            "stale or mismatched checkpoint, remove it manually and retry."
        )


def _validate_accelerator_allocation(
    torch_module: Any,
    tensor_parallel_size: int,
    strict_check: bool = True,
) -> None:
    """Validate that the accelerator allocation matches tensor_parallel_size."""
    assigned_gpu_ids = _get_ray_gpu_ids()
    visible_device_env = _get_visible_device_env()

    if tensor_parallel_size < 1:
        raise ValueError(
            "tensor_parallel_size must be at least 1, "
            f"got {tensor_parallel_size}"
        )

    if assigned_gpu_ids and len(assigned_gpu_ids) != tensor_parallel_size:
        message = (
            "Ray GPU allocation does not match tensor_parallel_size: "
            f"ray_gpu_ids={assigned_gpu_ids}, "
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"visible_device_env={visible_device_env}"
        )
        if strict_check:
            raise RuntimeError(message)
        logger.warning(message)

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "PyTorch reports no available GPU/DCU device. "
            f"ray_gpu_ids={assigned_gpu_ids}, "
            f"visible_device_env={visible_device_env}, "
            f"torch_version={torch_module.__version__}, "
            f"torch_hip_version={getattr(torch_module.version, 'hip', None)}"
        )

    visible_device_count = torch_module.cuda.device_count()
    if visible_device_count < tensor_parallel_size:
        raise RuntimeError(
            "The vLLM downloader cannot see enough GPU/DCU devices: "
            f"visible_device_count={visible_device_count}, "
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"ray_gpu_ids={assigned_gpu_ids}, "
            f"visible_device_env={visible_device_env}"
        )

    if (
        strict_check
        and assigned_gpu_ids
        and visible_device_count != len(assigned_gpu_ids)
    ):
        raise RuntimeError(
            "Ray allocated a set of GPU/DCU devices, but PyTorch sees a "
            "different number. This can make multiple Ray actors use the "
            "same physical device. "
            f"visible_device_count={visible_device_count}, "
            f"ray_gpu_ids={assigned_gpu_ids}, "
            f"visible_device_env={visible_device_env}"
        )

    logger.info(
        "Accelerator allocation validated: ray_gpu_ids=%s, "
        "visible_device_count=%s, hip_runtime=%s",
        assigned_gpu_ids,
        visible_device_count,
        getattr(torch_module.version, "hip", None),
    )


def _filter_supported_kwargs(
    callable_obj: Any, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Filter keyword arguments to those supported by the installed vLLM."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    supported_names = set(signature.parameters)
    filtered = {
        name: value for name, value in kwargs.items() if name in supported_names
    }
    ignored = sorted(set(kwargs) - set(filtered))
    if ignored:
        logger.info(
            "Ignoring vLLM arguments unsupported by this build: %s",
            ignored,
        )
    return filtered


def _save_serverless_llm_state(
    llm_writer: Any,
    model_path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> Any:
    """Save the model in ServerlessLLM format.

    vLLM V1 exposes the save entry point on the engine core.  Older or
    vendor-forked engines (e.g. Hygon DCU builds based on V0) do not have it,
    but expose ``apply_model`` which allows the sllm_store vLLM plugin to
    reach the model directly; use that as the fallback.
    """
    llm_engine = getattr(llm_writer, "llm_engine", None)

    # Preferred path: vLLM V1 engine core.
    engine_core = getattr(llm_engine, "engine_core", None)
    save_on_core = getattr(engine_core, "save_serverless_llm_state", None)
    if callable(save_on_core):
        result = save_on_core(path=model_path, pattern=pattern, max_size=max_size)
        return _resolve_maybe_remote_result(result)

    # Fallback: apply_model + sllm_store vllm plugin (V0-style engines).
    from sllm_store.vllm_loader import save_serverless_llm_state

    apply_model = getattr(llm_engine, "apply_model", None)
    if not callable(apply_model):
        raise RuntimeError(
            "The installed vLLM build exposes neither "
            "llm_engine.engine_core.save_serverless_llm_state nor "
            "llm_engine.apply_model; cannot save the model in "
            "ServerlessLLM format."
        )

    save_fn = partial(save_serverless_llm_state, output_path=str(model_path))
    result = apply_model(save_fn)
    result = _resolve_maybe_remote_result(result)

    if pattern is not None or max_size is not None:
        logger.warning(
            "This vLLM build does not support pattern/max_size when saving "
            "ServerlessLLM checkpoints; ignoring pattern=%r, max_size=%r",
            pattern,
            max_size,
        )

    logger.info("Saved ServerlessLLM rank paths: %s", result)
    return result


def _resolve_maybe_remote_result(result: Any) -> Any:
    """Resolve a Ray ObjectRef or awaitable returned by vLLM internals."""
    if isinstance(result, ray.ObjectRef):
        return ray.get(result)
    return result


def _ignore_weight_files(_directory: str, names: List[str]) -> List[str]:
    return [
        name
        for name in names
        if os.path.splitext(name)[1].lower() in _WEIGHT_FILE_SUFFIXES
    ]


def _copy_model_metadata(input_dir: str, model_path: str) -> None:
    """Copy config/tokenizer/code files without the original weight files."""
    os.makedirs(model_path, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if os.path.splitext(file_name)[1].lower() in _WEIGHT_FILE_SUFFIXES:
            continue

        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(model_path, file_name)

        if os.path.isdir(src_path):
            shutil.copytree(
                src_path,
                dest_path,
                dirs_exist_ok=True,
                ignore=_ignore_weight_files,
            )
        else:
            shutil.copy2(src_path, dest_path)


def _shutdown_vllm_writer(llm_writer: Any) -> None:
    """Best-effort cleanup for different vLLM engine implementations."""
    if llm_writer is None:
        return

    llm_engine = getattr(llm_writer, "llm_engine", None)
    shutdown_candidates = [
        ("llm_writer.shutdown", llm_writer, "shutdown"),
        ("llm_engine.shutdown", llm_engine, "shutdown"),
        (
            "llm_engine.shutdown_background_loop",
            llm_engine,
            "shutdown_background_loop",
        ),
    ]

    for method_name, target, attribute_name in shutdown_candidates:
        if target is None or not hasattr(target, attribute_name):
            continue
        try:
            getattr(target, attribute_name)()
            logger.info("Called %s", method_name)
            break
        except Exception as exc:
            logger.warning("Failed to call %s: %s", method_name, exc)


def _release_accelerator_memory(torch_module: Any) -> None:
    """Release CUDA/HIP/DTK memory through PyTorch's compatible API."""
    if not torch_module.cuda.is_available():
        return

    try:
        torch_module.cuda.synchronize()
    except Exception as exc:
        logger.warning("Failed to synchronize the accelerator runtime: %s", exc)

    try:
        torch_module.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Failed to empty the accelerator memory cache: %s", exc)


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        pretrained_model_name_or_path: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
        backend_config: Optional[Dict[str, Any]] = None,
    ):
        """Convert a model into the ServerlessLLM format on the assigned GPUs.

        The conversion runs in a Ray actor with ``num_gpus`` devices.  On
        ROCm/DTK the allocation is pinned into ROCR_VISIBLE_DEVICES before
        device use; output is published atomically from a staging dir.
        """
        import gc
        from tempfile import TemporaryDirectory, mkdtemp

        backend_config = backend_config or {}
        accelerator = str(
            backend_config.get(
                "accelerator", os.getenv("SLLM_ACCELERATOR", "auto")
            )
        ).lower()

        llm_writer = None
        torch_module = None
        model_path = None

        try:
            # vLLM reads these at import time; set only when configured.
            if backend_config.get("vllm_use_v1") is not None:
                os.environ["VLLM_USE_V1"] = str(
                    backend_config["vllm_use_v1"]
                )
            if backend_config.get("rocm_path"):
                os.environ.setdefault(
                    "ROCM_PATH", str(backend_config["rocm_path"])
                )

            import torch
            torch_module = torch

            if _is_rocm_accelerator(accelerator, torch):
                # Needed by some vLLM builds to unpickle older shards.
                os.environ.setdefault(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION", "1"
                )
                # mp workers avoid a nested Ray executor in this Ray actor.
                os.environ.setdefault(
                    "VLLM_WORKER_MULTIPROC_METHOD", "spawn"
                )
                _sync_rocm_visible_devices(_get_ray_gpu_ids())

            from huggingface_hub import snapshot_download
            from vllm import LLM

            tensor_parallel_size = int(tensor_parallel_size)
            if _is_rocm_accelerator(accelerator, torch):
                # Validate allocation on DCU only; CUDA keeps upstream.
                _validate_accelerator_allocation(
                    torch,
                    tensor_parallel_size,
                    strict_check=bool(
                        backend_config.get("strict_accelerator_check", True)
                    ),
                )

            # set the storage path
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join(storage_path, "vllm", model_name)

            if os.path.exists(model_path):
                # Reuse only a complete checkpoint; never touch existing dirs.
                _check_rank_layout(model_path, tensor_parallel_size)
                logger.info(f"{model_path} already exists")
                return

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Private staging dir + atomic rename: failures and concurrent
            # registrations cannot clobber each other's output.
            staging_path = mkdtemp(
                prefix=f".{model_name}.tmp-",
                dir=os.path.dirname(model_path),
            )
            committed = False
            try:
                with TemporaryDirectory() as cache_dir:
                    if os.path.exists(pretrained_model_name_or_path):
                        input_dir = os.path.abspath(
                            pretrained_model_name_or_path
                        )
                    else:
                        # download from huggingface
                        input_dir = snapshot_download(
                            pretrained_model_name_or_path,
                            cache_dir=cache_dir,
                            allow_patterns=[
                                "*.safetensors",
                                "*.bin",
                                "*.json",
                                "*.txt",
                            ],
                        )
                    logger.info(f"Loading model from {input_dir}")

                    # load models from the input directory
                    llm_writer = LLM(
                        **_filter_supported_kwargs(
                            LLM,
                            {
                                "model": input_dir,
                                "download_dir": input_dir,
                                "dtype": torch_dtype,
                                "tensor_parallel_size": tensor_parallel_size,
                                "num_gpu_blocks_override": 1,
                                "enforce_eager": True,
                                "max_model_len": 1,
                                "disable_log_stats": True,
                                "trust_remote_code": backend_config.get(
                                    "trust_remote_code", False
                                ),
                            },
                        )
                    )
                    # save the models in the ServerlessLLM format
                    _save_serverless_llm_state(
                        llm_writer, staging_path, pattern, max_size
                    )
                    # Copy the metadata files into the output directory
                    _copy_model_metadata(input_dir, staging_path)

                _check_rank_layout(staging_path, tensor_parallel_size)
                try:
                    os.rename(staging_path, model_path)
                    committed = True
                except OSError:
                    # Concurrent registration published first; reuse if OK.
                    if not os.path.isdir(model_path):
                        raise
                    _check_rank_layout(model_path, tensor_parallel_size)
                    logger.info(f"{model_path} already exists")
            finally:
                if not committed:
                    shutil.rmtree(staging_path, ignore_errors=True)

            logger.info(
                "Saved %s in ServerlessLLM format with TP=%s",
                model_name,
                tensor_parallel_size,
            )
        except Exception as e:
            logger.exception(
                "Failed to save %s for vllm backend: %s", model_name, e
            )
            raise RuntimeError(
                f"Failed to save {model_name} for vllm backend: {e}"
            ) from e
        finally:
            _shutdown_vllm_writer(llm_writer)
            llm_writer = None
            gc.collect()
            if torch_module is not None:
                _release_accelerator_memory(torch_module)
