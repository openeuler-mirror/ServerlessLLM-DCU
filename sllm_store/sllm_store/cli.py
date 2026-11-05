# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                         #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");               #
#  you may not use this file except in compliance with the License.              #
#  You may obtain a copy of the License at                                     #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#  Unless required by applicable law or agreed to in writing, software           #
#  distributed under the License is distributed on an "AS IS" BASIS,            #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
#  See the License for the specific language governing permissions and          #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import importlib
import logging
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from tempfile import TemporaryDirectory
from typing import Optional

import torch
import click

from sllm_store.server import serve
from sllm_store.logger import init_logger
from sllm_store.utils import to_num_bytes
from sllm_store.transformers import save_model, save_lora, load_model, load_lora
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


logger = init_logger(__name__)

_WEIGHT_SUFFIXES = {
    ".bin", ".pt", ".pth", ".safetensors", ".ckpt", ".h5", ".msgpack",
    ".onnx", ".gguf",
}


_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tekken.json",
)


def _has_tokenizer_files(model_path: Path) -> bool:
    """Return whether a saved model directory contains tokenizer assets."""
    return model_path.is_dir() and any(
        (model_path / filename).is_file() for filename in _TOKENIZER_FILES
    )


def resolve_storage_path(cli_path: Optional[str]) -> str:
    """Resolve storage path: CLI > STORAGE_PATH > ~/models/."""
    if cli_path is not None:
        return os.path.expanduser(cli_path)
    env_path = os.getenv("STORAGE_PATH")
    if env_path:
        return os.path.expanduser(env_path)
    return os.path.expanduser("~/models/")


def _vllm_model_path(storage_path: str, model_name: str) -> Path:
    """Allow names such as org/model, but reject escaping or aliased paths."""
    if (
        not model_name
        or "\\" in model_name
        or any(ord(char) < 32 for char in model_name)
        or any(part in ("", ".", "..") for part in model_name.split("/"))
    ):
        raise ValueError(
            "model_name must be a relative name such as 'qwen7b' or "
            "'Qwen/Qwen2.5-7B', without empty, '.' or '..' components"
        )
    root = Path(storage_path).expanduser().resolve()
    target = root
    for part in model_name.split("/"):
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Model path must not contain symlinks: {target}")
    resolved = target.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Model path escapes storage directory: {model_name}")
    return target


def _transformers_output_name(model_identifier: str) -> str:
    """Convert a Transformers model/adapter identifier to a safe storage key.

    Hugging Face IDs such as ``org/model`` keep their namespace. If the caller
    supplied an absolute local path as ``--model`` or ``--adapter-name``, only
    its final directory name is used as the Store key.
    """
    if not model_identifier:
        raise ValueError("Model or adapter name must not be empty")

    if "\\" in model_identifier or any(
        ord(char) < 32 for char in model_identifier
    ):
        raise ValueError("Model or adapter name contains an invalid character")

    candidate = Path(model_identifier).expanduser()
    if candidate.is_absolute():
        output_name = candidate.name
    else:
        output_name = model_identifier[2:] if model_identifier.startswith("./") else model_identifier

    if (
        not output_name
        or any(part in ("", ".", "..") for part in output_name.split("/"))
    ):
        raise ValueError(
            "Model or adapter name must be a relative name such as 'qwen7b' "
            "or 'Qwen/Qwen2.5-7B', or an absolute local source path"
        )
    return output_name


def _transformers_model_path(storage_path: str, model_identifier: str) -> Path:
    """Resolve a Transformers model/adapter key safely under storage_path."""
    output_name = _transformers_output_name(model_identifier)
    return _vllm_model_path(storage_path, output_name)


def _copy_vllm_metadata(input_dir: Path, output_dir: Path) -> None:
    """Copy metadata without following links outside this model/cache repo."""
    input_dir = input_dir.resolve(strict=True)
    allowed_roots = [input_dir]
    # HF snapshots link to ../../blobs/<hash> in the same cached repository.
    if (
        input_dir.parent.name == "snapshots"
        and input_dir.parent.parent.name.startswith("models--")
    ):
        blobs = input_dir.parent.parent / "blobs"
        if blobs.is_dir() and not blobs.is_symlink():
            allowed_roots.append(blobs)

    def copy_file(source: Path, destination: Path) -> None:
        resolved = source.resolve(strict=True)
        for root in allowed_roots:
            try:
                relative = resolved.relative_to(root)
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Metadata link escapes the model and its HF blob directory: "
                f"{source} -> {resolved}"
            )

        # Traverse via directory FDs and O_NOFOLLOW so swapping a link after
        # resolution cannot redirect the file open outside the allowed root.
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(root, directory_flags)
        source_fd = None
        try:
            if not relative.parts:
                raise ValueError(f"Unsupported metadata entry: {source}")
            for component in relative.parts[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            source_fd = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            file_stat = os.fstat(source_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"Unsupported metadata entry: {source}")
            with os.fdopen(source_fd, "rb") as reader:
                source_fd = None  # fdopen owns the descriptor now.
                with destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
            os.chmod(destination, stat.S_IMODE(file_stat.st_mode) & 0o777)
            os.utime(destination, ns=(file_stat.st_atime_ns, file_stat.st_mtime_ns))
        finally:
            if source_fd is not None:
                os.close(source_fd)
            os.close(directory_fd)

    def copy_directory(source_dir: Path, destination_dir: Path) -> None:
        with os.scandir(source_dir) as entries:
            for entry in entries:
                source = Path(entry.path)
                name = entry.name
                if (
                    name in {".git", ".cache", "__pycache__"}
                    or source.suffix.lower() in _WEIGHT_SUFFIXES
                    or name.lower().endswith((
                        ".safetensors.index.json", ".bin.index.json"
                    ))
                    or (entry.is_symlink() and entry.is_dir())
                ):
                    continue
                destination = destination_dir / name
                if os.path.lexists(destination):
                    raise FileExistsError(
                        f"Model metadata would overwrite saved output: {destination}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    destination.mkdir()
                    copy_directory(source, destination)
                else:
                    copy_file(source, destination)

    copy_directory(input_dir, output_dir)


def _publish_vllm_checkpoint(
    staging_path: Path, storage_path: str, model_name: str
) -> Path:
    """Serialize CLI publication in the parent directory on Linux/POSIX."""
    import fcntl

    model_path = _vllm_model_path(storage_path, model_name)
    parent_fd = os.open(model_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        # Check again under the lock: another CLI saver may have finished.
        model_path = _vllm_model_path(storage_path, model_name)
        if os.path.lexists(model_path):
            raise FileExistsError(
                f"Model output already exists; refusing to overwrite: {model_path}"
            )
        staging_path.rename(model_path)
    finally:
        os.close(parent_fd)
    return model_path


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        storage_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ):
        import gc
        from functools import partial
        from tempfile import TemporaryDirectory

        storage_path = resolve_storage_path(storage_path)
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        model_path = _vllm_model_path(storage_path, model_name)
        if os.path.lexists(model_path):
            raise FileExistsError(
                f"Model output already exists; refusing to overwrite: {model_path}"
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Set these before importing vLLM. Scope the changes to this save call;
        # spawned workers inherit them. The callback requires trusted local RPC.
        env_names = (
            "VLLM_WORKER_MULTIPROC_METHOD",
            "VLLM_ALLOW_INSECURE_SERIALIZATION",
        )
        previous_env = {name: os.environ.get(name) for name in env_names}
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        def _run_writer(input_dir: Path, output_dir: Path):
            llm_writer = None
            try:
                # Keep the original in-process initialization and parameters.
                llm_writer = LLM(
                    model=str(input_dir),
                    download_dir=str(input_dir),
                    dtype=torch_dtype,
                    tensor_parallel_size=tensor_parallel_size,
                    num_gpu_blocks_override=1,
                    enforce_eager=True,
                    max_model_len=1,
                    enable_chunked_prefill=False,
                    disable_log_stats=True,
                    disable_custom_all_reduce=True,
                )
                save_results = llm_writer.llm_engine.apply_model(
                    partial(
                        save_serverless_llm_state, output_path=str(output_dir)
                    )
                )
                logger.info(
                    f"Worker save results (temporary paths): {save_results}"
                )
                _copy_vllm_metadata(input_dir, output_dir)
            finally:
                # Best-effort cleanup; do not hide the original save error.
                # This does not guarantee that all vLLM worker processes stop.
                del llm_writer
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.warning("GPU cache cleanup failed", exc_info=True)

        try:
            from huggingface_hub import snapshot_download
            from vllm import LLM
            from sllm_store.vllm_loader import save_serverless_llm_state

            with TemporaryDirectory() as cache_dir:
                if local_model_path is None:
                    input_dir = snapshot_download(
                        model_name,
                        cache_dir=cache_dir,
                        allow_patterns=[
                            "*.safetensors", "*.bin", "*.json", "*.txt",
                            "*.model", "*.tiktoken", "*.vocab", "*.merges",
                            "*.jinja", "*.jinja2", "*.yaml", "*.yml", "*.py",
                            "vocab", "merges", "tokenizer*",
                        ],
                    )
                else:
                    input_dir = local_model_path
                input_dir = Path(input_dir).expanduser().resolve(strict=True)
                if not input_dir.is_dir():
                    raise NotADirectoryError(f"Invalid model directory: {input_dir}")
                if input_dir == model_path or input_dir in model_path.parents:
                    raise ValueError(
                        "Output model directory must not be inside the source "
                        "model directory"
                    )

                # Same filesystem as the destination, so the final rename is
                # atomic. Only this private temporary directory is cleaned up.
                with TemporaryDirectory(
                    prefix=f".{model_path.name}.saving-",
                    dir=model_path.parent,
                ) as temp_dir:
                    output_dir = Path(temp_dir) / "checkpoint"
                    output_dir.mkdir()
                    _run_writer(input_dir, output_dir)
                    final_path = _publish_vllm_checkpoint(
                        output_dir, storage_path, model_name
                    )
                    logger.info(f"Saved ServerlessLLM checkpoint: {final_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save {model_name} for vllm backend: {exc}"
            ) from exc
        finally:
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@click.group()
def cli():
    """sllm-store CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8073, help="Port")
@click.option(
    "--storage-path",
    default=None,
    help="Storage path (default: $STORAGE_PATH or ~/models/)",
)
@click.option("--num-thread", default=4, help="Number of I/O threads")
@click.option(
    "--chunk-size", default="32MB", help="Chunk size, e.g., 4KB, 1MB, 1GB"
)
@click.option(
    "--mem-pool-size",
    default="4GB",
    help="Memory pool size, e.g., 1GB, 4GB, 1TB",
)
@click.option(
    "--disk-size", default="128GB", help="Disk size, e.g., 1GB, 4GB, 1TB"
)
@click.option(
    "--registration-required",
    default=False,
    help="Require registration before loading model",
)
def start(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    disk_size,
    registration_required,
):
    """Start the gRPC server."""
    storage_path = resolve_storage_path(storage_path)
    chunk_size = to_num_bytes(chunk_size)
    mem_pool_size = to_num_bytes(mem_pool_size)
    try:
        logger.info("Starting gRPC server")
        asyncio.run(
            serve(
                host=host,
                port=port,
                storage_path=storage_path,
                num_thread=num_thread,
                chunk_size=chunk_size,
                mem_pool_size=mem_pool_size,
                # disk size is not used
                # disk_size=disk_size,
                registration_required=registration_required,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)


@cli.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub or a local model path",
)
@click.option("--backend", type=str, required=True, help="Backend")
@click.option("--adapter-name", type=str, help="Name or path of the LoRA adapter")
@click.option(
    "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
)
@click.option(
    "--local-model-path", type=str, help="Local path to the model snapshot"
)
@click.option(
    "--storage-path",
    default=None,
    help="Local path to save the model (default: $STORAGE_PATH or ~/models/)",
)
def save(
    model_name,
    backend,
    adapter_name,
    tensor_parallel_size,
    local_model_path,
    storage_path,
):
    """
    Saves a model to the sllm-store's storage.

    This command is for adding new models to the sllm-store's local storage.
    """
    storage_path = resolve_storage_path(storage_path)
    logger.info(
        f"Saving model {adapter_name if adapter_name else model_name} "
        f"to {storage_path}"
    )
    try:
        if backend == "vllm":
            downloader = VllmModelDownloader()
            vllm_storage_path = os.path.join(storage_path, "vllm")
            downloader.download_vllm_model(
                model_name,
                "float16",
                tensor_parallel_size=tensor_parallel_size,
                storage_path=vllm_storage_path,
                local_model_path=local_model_path,
            )
        elif backend == "transformers":
            # The model identifier is the Store key; local_model_path is the
            # source directory. Never use an absolute source path as output.
            output_identifier = adapter_name or model_name
            model_path = _transformers_model_path(
                storage_path, output_identifier
            )
            if os.path.lexists(model_path):
                raise FileExistsError(
                    f"Model output already exists; refusing to overwrite: {model_path}"
                )

            model_source = local_model_path or model_name
            source_candidate = Path(model_source).expanduser()
            source_dir = None
            if local_model_path is not None or source_candidate.is_dir():
                source_dir = source_candidate.resolve(strict=True)
                if not source_dir.is_dir():
                    raise NotADirectoryError(
                        f"Invalid model source directory: {source_dir}"
                    )
                model_source = str(source_dir)

            # Reject both nesting directions to avoid writing any converted
            # files into the source model tree (or replacing its parent).
            resolved_output = model_path.resolve()
            if source_dir is not None and (
                source_dir == resolved_output
                or source_dir in resolved_output.parents
                or resolved_output in source_dir.parents
            ):
                raise ValueError(
                    "Source model directory and output model directory must not overlap"
                )

            model_path.parent.mkdir(parents=True, exist_ok=True)
            # Stage the complete output on the same filesystem, then publish it
            # atomically. A failed conversion leaves no partial target directory.
            with TemporaryDirectory(
                prefix=f".{model_path.name}.saving-",
                dir=model_path.parent,
            ) as temporary_dir:
                staging_path = Path(temporary_dir) / "checkpoint"
                staging_path.mkdir()

                if adapter_name:
                    config = AutoConfig.from_pretrained(
                        model_source,
                        trust_remote_code=True,
                    )
                    config.torch_dtype = torch.float16
                    module = importlib.import_module("transformers")
                    hf_model_cls = module.AutoModelForCausalLM
                    base_model = hf_model_cls.from_config(
                        config,
                        trust_remote_code=True,
                    ).to(config.torch_dtype)
                    model = PeftModel.from_pretrained(base_model, adapter_name)
                    save_lora(model, str(staging_path))
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_source, torch_dtype=torch.float16
                    )
                    save_model(model, str(staging_path))

                    # Keep tokenizer assets with the converted checkpoint.
                    # This lets load-time example inference use the Store
                    # directory instead of treating the Store key as an HF ID.
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_source,
                        trust_remote_code=True,
                    )
                    tokenizer.save_pretrained(str(staging_path))

                _publish_vllm_checkpoint(
                    staging_path, storage_path, _transformers_output_name(output_identifier)
                )
        else:
            logger.error(f"Unsupported backend '{backend}'")
            sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to save model {model_name}: {exc}", exc_info=True)
        sys.exit(1)
    logger.info(
        f"Model {adapter_name if adapter_name else model_name} "
        f"saved successfully to {storage_path}"
    )


@cli.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub",
)
@click.option("--backend", type=str, required=True, help="Backend")
@click.option("--adapter-name", type=str, help="Name of the LoRA adapter")
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Tensor parallel size the vLLM checkpoint was saved with",
)
@click.option(
    "--precision",
    type=click.Choice(["int8", "fp4", "nf4"]),
    help="Precision of quantized model.",
)
@click.option(
    "--storage-path",
    type=str,
    default=None,
    help="Local path where model is saved (default: $STORAGE_PATH or ~/models/)",
)
def load(
    model_name,
    backend,
    adapter_name,
    tensor_parallel_size,
    precision,
    storage_path,
):
    """
    Loads a model from the sllm-store's storage.

    This command is for loading new models from the sllm-store's local storage.
    """
    storage_path = resolve_storage_path(storage_path)
    logger.info(
        f"Loading model {adapter_name if adapter_name else model_name} "
        f"from {storage_path}"
    )
    quantization_config = None
    if precision:
        if precision == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "fp4":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif precision == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4"
            )
    try:
        start_load_time = time.time()
        if backend == "vllm":
            if tensor_parallel_size < 1:
                raise ValueError("tensor_parallel_size must be at least 1")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            model_full_path = _vllm_model_path(
                os.path.join(storage_path, "vllm"), model_name
            )
            from vllm import LLM

            # Preserve the existing loader registration and extra config.
            llm = LLM(
                model=str(model_full_path),
                load_format="serverless_llm",
                dtype="float16",
                tensor_parallel_size=tensor_parallel_size,
                model_loader_extra_config={"storage_path": storage_path},
            )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )
            example_inferences("vllm", model=llm)
        elif backend == "transformers":
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                torch.ones(1).to(f"cuda:{i}")
                torch.cuda.synchronize()
            # A save invoked with an absolute local --model path stores under
            # its basename. Resolve the same key when loading from Store.
            store_model_name = _transformers_output_name(model_name)
            if adapter_name:
                store_adapter_name = _transformers_output_name(adapter_name)
                model = load_model(
                    store_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    storage_path=storage_path,
                    fully_parallel=True,
                )
                model = load_lora(
                    model,
                    adapter_name,
                    adapter_path=store_adapter_name,
                    device_map="auto",
                    storage_path=storage_path,
                    torch_dtype=torch.float16,
                )
            else:
                model = load_model(
                    store_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    storage_path=storage_path,
                    fully_parallel=True,
                    quantization_config=quantization_config,
                )
            logger.info(
                f"Model loading time: {time.time() - start_load_time:.2f}s"
            )
            example_inferences(
                "transformers",
                model=model,
                model_name=model_name,
                adapter_name=adapter_name,
                storage_path=storage_path,
                store_model_name=store_model_name,
            )
        else:
            logger.error(f"Unsupported backend '{backend}'")
            sys.exit(1)
    except Exception as exc:
        logger.error(
            f"Failed to load model or perform inference: {exc}", exc_info=True
        )
        sys.exit(1)
    logger.info(
        f"Model {adapter_name if adapter_name else model_name} "
        f"loaded successfully from {storage_path}"
    )


def example_inferences(
    backend,
    model=None,
    model_name=None,
    adapter_name=None,
    storage_path=None,
    store_model_name=None,
):
    if backend == "vllm":
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = model.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    elif backend == "transformers":
        tokenizer_source = None
        local_files_only = False

        # Prefer tokenizer files saved alongside the converted Store model.
        if storage_path is not None and store_model_name is not None:
            stored_model_path = _transformers_model_path(
                storage_path, store_model_name
            )
            if _has_tokenizer_files(stored_model_path):
                tokenizer_source = str(stored_model_path)

        # Backward compatibility: accept an explicitly supplied local source
        # directory when it is still available.
        source_candidate = Path(model_name).expanduser()
        if tokenizer_source is None and source_candidate.is_dir():
            tokenizer_source = str(source_candidate.resolve())

        # If this is an HF namespace/repository ID, preserve the previous
        # online fallback. A bare Store alias is checked locally only so it
        # cannot trigger a network request such as huggingface.co/qwen7b.
        if tokenizer_source is None:
            tokenizer_source = model_name
            local_files_only = (
                "/" not in model_name
                and not source_candidate.is_absolute()
                and not source_candidate.is_dir()
            )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(
            "cuda"
        )
        generate_kwargs = {}
        if adapter_name:
            generate_kwargs["adapter_names"] = [adapter_name]
        outputs = model.generate(**inputs, **generate_kwargs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        logger.error(f"Unsupported backend '{backend}' for example inferences")
        sys.exit(1)


# Entry point for the 'sllm-store' CLI
def main():
    cli()

