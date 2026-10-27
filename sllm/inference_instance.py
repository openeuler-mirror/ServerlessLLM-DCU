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

import os
from copy import deepcopy
from typing import Any, Dict, Mapping

import ray

from sllm.logger import init_logger

logger = init_logger(__name__)

# Accelerator runtime vars propagated to the model actor (ROCm/DTK);
# visibility vars excluded (Ray controls them), LD_LIBRARY_PATH kept for DTK.
_RUNTIME_ENV_PASSTHROUGH = (
    "ROCM_PATH",
    "HIP_PATH",
    "DTK_PATH",
    "LD_LIBRARY_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "VLLM_LOGGING_LEVEL",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_USE_V1",
)

# Per-model overrides (backend_config) for the actor environment; they win
# over the worker environment but never over an explicit runtime_env.
_CONFIGURABLE_ENV_KEYS = ("VLLM_USE_V1", "ROCM_PATH")


def _prepare_startup_config(
    backend: str,
    backend_config: Mapping[str, Any],
    startup_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Propagate accelerator runtime variables to the model actor options.

    No-op when nothing needs propagating.  Precedence: explicit
    runtime_env > backend_config overrides > worker environment.
    """
    if backend != "vllm":
        return dict(startup_config or {})

    backend_config = backend_config or {}
    env_vars: Dict[str, str] = {}

    for name in _RUNTIME_ENV_PASSTHROUGH:
        value = os.getenv(name)
        if value:
            env_vars[name] = value

    for key in _CONFIGURABLE_ENV_KEYS:
        value = backend_config.get(key)
        if value is not None:
            env_vars[key] = str(value)

    if not env_vars:
        # Nothing to propagate; keep upstream options untouched.
        return dict(startup_config or {})

    prepared = deepcopy(dict(startup_config or {}))
    runtime_env = deepcopy(dict(prepared.get("runtime_env") or {}))
    explicit_env_vars = deepcopy(dict(runtime_env.get("env_vars") or {}))
    # Explicitly configured runtime_env entries always win.
    runtime_env["env_vars"] = {**env_vars, **explicit_env_vars}
    prepared["runtime_env"] = runtime_env
    logger.info(
        "Propagating accelerator runtime variables to instance actor: %s",
        sorted(env_vars),
    )
    return prepared


@ray.remote
def start_instance(
    instance_id, backend, model_name, backend_config, startup_config
):
    logger.info(
        "Starting instance %s with backend=%s, model=%s",
        instance_id,
        backend,
        model_name,
    )

    prepared_startup_config = _prepare_startup_config(
        backend, backend_config or {}, startup_config or {}
    )

    # The backend import stays inside this worker task so that vLLM and the
    # accelerator runtime are imported on the worker node, not in the
    # controller.
    if backend == "vllm":
        from sllm.backends import VllmBackend

        model_backend_cls = VllmBackend
    elif backend == "dummy":
        from sllm.backends import DummyBackend

        model_backend_cls = DummyBackend
    elif backend == "transformers":
        from sllm.backends import TransformersBackend

        model_backend_cls = TransformersBackend
    else:
        logger.error(f"Unknown backend: {backend}")
        raise ValueError(f"Unknown backend: {backend}")

    model_actor_cls = ray.remote(model_backend_cls)

    return model_actor_cls.options(
        name=instance_id,
        **prepared_startup_config,
        max_concurrency=10,
        lifetime="detached",
    ).remote(model_name, backend_config)
