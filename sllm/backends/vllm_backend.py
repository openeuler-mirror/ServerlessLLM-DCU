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
import asyncio
import gc
import inspect
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Set, Union, cast

import ray
import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt
from vllm.utils import Counter

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)

logger = logging.getLogger("ray")

# Device visibility variables used by CUDA and ROCm/DTK (Hygon DCU) runtimes.
_VISIBLE_DEVICE_ENV_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

def _is_hip_runtime() -> bool:
    """Whether PyTorch is built on HIP (AMD GPUs / Hygon DCUs)."""
    return getattr(torch.version, "hip", None) is not None


def _get_async_engine_arg_names() -> Set[str]:
    """Return AsyncEngineArgs fields, tolerating non-dataclass vLLM forks."""
    try:
        return {field.name for field in fields(AsyncEngineArgs)}
    except (TypeError, ValueError):
        return set(inspect.signature(AsyncEngineArgs).parameters)

def _get_ray_gpu_ids() -> List[str]:
    """Read the logical GPU/DCU IDs Ray assigned to this model actor."""
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


def _release_accelerator_memory() -> None:
    """Release cached accelerator memory; ROCm/DTK also uses torch.cuda."""
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception as exc:
        logger.warning("Accelerator synchronize failed during shutdown: %s", exc)

    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Accelerator cache cleanup failed during shutdown: %s", exc)


async def _call_shutdown_method(obj: Any, method_name: str) -> bool:
    """Call a possibly async shutdown method on a vLLM object."""
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return False

    result = method()
    if inspect.isawaitable(result):
        await result
    return True


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        # Get the accelerator IDs assigned to the current Ray actor.
        assigned_gpu_ids = _get_ray_gpu_ids()
        is_hip = _is_hip_runtime()

        # Default to Ray's allocated device count for tensor parallelism on HIP.
        tensor_parallel_size: Optional[int] = None
        if is_hip:
            tensor_parallel_size = int(
                backend_config.get(
                    "tensor_parallel_size",
                    len(assigned_gpu_ids) if assigned_gpu_ids else 1,
                )
            )
            if tensor_parallel_size < 1:
                raise ValueError(
                    "tensor_parallel_size must be >= 1, "
                    f"got {tensor_parallel_size}"
                )

            strict_accelerator_check = bool(
                backend_config.get("strict_accelerator_check", True)
            )
            if assigned_gpu_ids and (
                len(assigned_gpu_ids) != tensor_parallel_size
            ):
                message = (
                    "Ray GPU allocation does not match vLLM tensor parallel "
                    f"size: ray_gpu_ids={assigned_gpu_ids}, "
                    f"tensor_parallel_size={tensor_parallel_size}, "
                    f"visible_device_env={_get_visible_device_env()}. "
                    "Set model_config.num_gpus equal to "
                    "backend_config.tensor_parallel_size."
                )
                if strict_accelerator_check:
                    raise RuntimeError(message)
                logger.warning(message)

            logger.info(
                "Accelerator assignment: ray_gpu_ids=%s, "
                "tensor_parallel_size=%s, hip_runtime=%s",
                assigned_gpu_ids,
                tensor_parallel_size,
                is_hip,
            )

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)

        # Use eager execution and disable prefix caching by default on HIP.
        self.enforce_eager = backend_config.get(
            "enforce_eager", is_hip
        )
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", not is_hip
        )
        self.task = backend_config.get("task", "auto")
        async_engine_fields = _get_async_engine_arg_names()

        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            pretrained_path = backend_config.get(
                "pretrained_model_name_or_path"
            )
            if not pretrained_path:
                raise ValueError(
                    "pretrained_model_name_or_path is required when "
                    "load_format is explicitly configured"
                )
            filtered_engine_config["model"] = pretrained_path
        else:
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        # Use multiprocessing for tensor parallelism on HIP runtimes.
        if is_hip:
            filtered_engine_config["tensor_parallel_size"] = tensor_parallel_size
            filtered_engine_config["distributed_executor_backend"] = (
                backend_config.get("distributed_executor_backend", "mp")
            )

        # Disable custom all-reduce by default on HIP runtimes.
        if is_hip:
            filtered_engine_config["disable_custom_all_reduce"] = (
                backend_config.get("disable_custom_all_reduce", True)
            )

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return

            # Log device visibility right before creating the engine to make
            # accelerator allocation problems easy to diagnose.
            logger.info(
                "Initializing vLLM backend: ray_gpu_ids=%s, "
                "visible_device_env=%s",
                _get_ray_gpu_ids(),
                _get_visible_device_env(),
            )
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # TODO stream results

        # Non-stream case
        final_output = None
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

        return process_output(final_output, model_name)

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        engine = self.engine
        if engine is not None:
            # Abort all requests.
            requests = await self.request_trace.return_all_request_ids()
            tasks = [engine.abort(request_id) for request_id in requests]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Different vLLM versions expose different shutdown hooks; call
            # them opportunistically and fall back to garbage collection.
            try:
                await _call_shutdown_method(
                    engine, "shutdown_background_loop"
                )
            except Exception as exc:
                logger.warning(
                    "vLLM background-loop shutdown failed: %s", exc
                )

        self.engine = None
        del engine
        gc.collect()

        # ROCm/DTK PyTorch builds also expose the accelerator via torch.cuda.
        _release_accelerator_memory()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)
