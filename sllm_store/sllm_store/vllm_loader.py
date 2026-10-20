# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the ServerlessLLM project

from __future__ import annotations

import collections
import gc
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
try:
    from vllm.config.load import LoadConfig
except ImportError:  # Compatibility with vendor-customized vLLM builds.
    from vllm.config import LoadConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

_FORMAT_VERSION = 2
_MANIFEST_NAME = "serverless_llm_manifest.json"
_RANK_META_NAME = "serverless_llm_rank.json"
_TENSOR_INDEX_NAME = "tensor_index.json"
_MAX_NATIVE_SIZE = (1 << 63) - 1
# Match the dtypes supported by the native restore_tensors implementation.
_DTYPE_ELEMENT_SIZES = {
    "torch.float16": 2, "torch.bfloat16": 2,
    "torch.float32": 4, "torch.float64": 8,
    "torch.int8": 1, "torch.uint8": 1,
    "torch.int16": 2, "torch.int32": 4, "torch.int64": 8,
}


@dataclass(frozen=True)
class _ParallelLayout:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int

    @property
    def model_parallel_rank(self) -> int:
        # vLLM lays out model-parallel workers with TP as the inner dimension.
        # This remains rank_{tp_rank} when PP=1, preserving old checkpoints.
        return self.pp_rank * self.tp_size + self.tp_rank

    @property
    def model_parallel_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def rank_dir_name(self) -> str:
        return f"rank_{self.model_parallel_rank}"


@dataclass(frozen=True)
class _AliasSpec:
    owner_name: str
    size: tuple[int, ...]
    stride: tuple[int, ...]
    relative_storage_offset: int


class ServerlessLLMLoader(BaseModelLoader):
    """Load vLLM runtime-state shards through ServerlessLLM Store.

    The checkpoint is a runtime-format checkpoint: every model-parallel worker
    saves and restores its own already-sharded vLLM state_dict. The loader binds
    restored tensors directly to the model parameters/buffers and keeps the
    ServerlessLLM backing allocation alive for the lifetime of the model.

    Supported by this DTK/vLLM-0.11 adapter:
      * tensor parallelism and pipeline parallelism;
      * data-parallel replicas reusing the same model-parallel shards;
      * unquantized runtime states.

    Quantized runtime states are intentionally rejected because many vLLM
    quantizers mutate/repack weights in process_weights_after_loading(); a
    checkpoint format must record that processing stage before reloading them
    safely.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else dict(load_config.model_loader_extra_config)
        )

        storage_path = extra_config.pop("storage_path", None)
        self.storage_root = Path(
            storage_path
            or os.environ.get("STORAGE_PATH", "./models")
        ).expanduser().resolve()

        self.strict = self._pop_bool(extra_config, "strict", True)
        self.release_init_weights = self._pop_bool(
            extra_config, "release_init_weights", True
        )
        self.empty_cache_before_restore = self._pop_bool(
            extra_config, "empty_cache_before_restore", True
        )
        self.allow_legacy_checkpoint = self._pop_bool(
            extra_config, "allow_legacy_checkpoint", True
        )

        if extra_config:
            raise ValueError(
                "Unexpected model_loader_extra_config keys for "
                f"serverless_llm: {sorted(extra_config)}"
            )

    @staticmethod
    def _pop_bool(config: dict[str, Any], key: str, default: bool) -> bool:
        value = config.pop(key, default)
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _parallel_layout() -> _ParallelLayout:
        tp_group = get_tp_group()
        pp_group = get_pp_group()
        return _ParallelLayout(
            tp_rank=tp_group.rank_in_group,
            tp_size=tp_group.world_size,
            pp_rank=pp_group.rank_in_group,
            pp_size=pp_group.world_size,
        )

    @staticmethod
    def _filter_subtensors(
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Keep one deterministic representative for overlapping storages.

        This selection intentionally matches the earlier DTK adapter so that
        checkpoints already produced by that adapter remain loadable.
        """
        groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for name, tensor in tensors.items():
            if tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            groups[(tensor.device, storage.data_ptr())].append((name, tensor))

        result: dict[str, torch.Tensor] = {}
        for group in groups.values():
            for name, tensor in group:
                if not tensor.is_contiguous():
                    result[name] = tensor
                    continue

                start = tensor.storage_offset() * tensor.element_size()
                end = start + tensor.numel() * tensor.element_size()
                covered = False

                for other_name, other in group:
                    if other_name == name or not other.is_contiguous():
                        continue

                    other_start = (
                        other.storage_offset() * other.element_size()
                    )
                    other_end = (
                        other_start
                        + other.numel() * other.element_size()
                    )
                    strictly_larger = other_start < start or end < other_end
                    same_range_with_preferred_name = (
                        other_start == start
                        and other_end == end
                        and other_name > name
                    )
                    if (
                        other_start <= start
                        and end <= other_end
                        and (
                            strictly_larger
                            or same_range_with_preferred_name
                        )
                    ):
                        covered = True
                        break

                if not covered:
                    result[name] = tensor

        return result

    @staticmethod
    def _build_alias_plan(
        state_vars: Mapping[str, torch.Tensor],
        owner_names: set[str],
    ) -> dict[str, _AliasSpec]:
        """Describe omitted tied/subtensor state entries as owner views."""
        groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for name, tensor in state_vars.items():
            if tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            groups[(tensor.device, storage.data_ptr())].append((name, tensor))

        aliases: dict[str, _AliasSpec] = {}
        for group in groups.values():
            owners = [(n, t) for n, t in group if n in owner_names]
            for name, tensor in group:
                if name in owner_names:
                    continue

                if not tensor.is_contiguous():
                    raise RuntimeError(
                        "ServerlessLLM runtime checkpoints currently require "
                        f"state tensor {name!r} to be contiguous."
                    )

                tensor_start = tensor.storage_offset()
                tensor_end = tensor_start + tensor.numel()
                selected: tuple[str, torch.Tensor] | None = None

                for owner_name, owner in owners:
                    if owner.dtype != tensor.dtype or not owner.is_contiguous():
                        continue
                    owner_start = owner.storage_offset()
                    owner_end = owner_start + owner.numel()
                    if owner_start <= tensor_start and tensor_end <= owner_end:
                        selected = (owner_name, owner)
                        break

                if selected is None:
                    raise RuntimeError(
                        "Unable to reconstruct shared state tensor "
                        f"{name!r} from the saved ServerlessLLM owners."
                    )

                owner_name, owner = selected
                aliases[name] = _AliasSpec(
                    owner_name=owner_name,
                    size=tuple(tensor.size()),
                    stride=tuple(tensor.stride()),
                    relative_storage_offset=(
                        tensor.storage_offset() - owner.storage_offset()
                    ),
                )

        return aliases

    def _model_root(self, model_config: ModelConfig) -> Path:
        model_weights = getattr(model_config, "model_weights", None)
        root = Path(model_weights or model_config.model).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"ServerlessLLM model directory does not exist: {root}"
            )
        return root

    def _model_id_for_rank(
        self,
        model_root: Path,
        layout: _ParallelLayout,
    ) -> tuple[str, Path]:
        canonical_rank_path = model_root / layout.rank_dir_name
        rank_path = canonical_rank_path

        # Legacy checkpoints used rank_{tp_rank}. This is identical to the new
        # layout when PP=1. Never use the legacy layout for PP>1 because ranks
        # from different pipeline stages would collide.
        if (
            not rank_path.is_dir()
            and self.allow_legacy_checkpoint
            and layout.pp_size == 1
        ):
            legacy_path = model_root / f"rank_{layout.tp_rank}"
            if legacy_path.is_dir():
                rank_path = legacy_path

        rank_path = rank_path.resolve()
        if not rank_path.is_dir():
            raise FileNotFoundError(
                "Missing ServerlessLLM rank directory. Expected "
                f"{canonical_rank_path} for TP={layout.tp_size}, "
                f"PP={layout.pp_size}, tp_rank={layout.tp_rank}, "
                f"pp_rank={layout.pp_rank}."
            )

        tensor_index = rank_path / _TENSOR_INDEX_NAME
        if not tensor_index.is_file():
            raise FileNotFoundError(
                f"Missing ServerlessLLM tensor index: {tensor_index}"
            )

        try:
            model_id = rank_path.relative_to(self.storage_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"The model rank path {rank_path} must be inside "
                f"STORAGE_PATH={self.storage_root}."
            ) from exc

        return model_id, rank_path

    def _validate_manifest(
        self,
        model_root: Path,
        layout: _ParallelLayout,
    ) -> None:
        manifest_path = model_root / _MANIFEST_NAME
        if not manifest_path.is_file():
            if self.allow_legacy_checkpoint:
                logger.warning(
                    "Loading a legacy ServerlessLLM checkpoint without %s. "
                    "TP/PP compatibility will be inferred from rank files.",
                    _MANIFEST_NAME,
                )
                return
            raise FileNotFoundError(
                f"Missing ServerlessLLM manifest: {manifest_path}"
            )

        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        version = int(manifest.get("format_version", 0))
        if version > _FORMAT_VERSION:
            raise RuntimeError(
                "Checkpoint format is newer than this loader: "
                f"checkpoint={version}, loader={_FORMAT_VERSION}."
            )

        saved_tp = int(manifest.get("tensor_parallel_size", 1))
        saved_pp = int(manifest.get("pipeline_parallel_size", 1))
        if saved_tp != layout.tp_size or saved_pp != layout.pp_size:
            raise RuntimeError(
                "ServerlessLLM checkpoint parallelism does not match the "
                f"current vLLM engine: saved TP={saved_tp}, PP={saved_pp}; "
                f"current TP={layout.tp_size}, PP={layout.pp_size}. "
                "Save the runtime checkpoint again with the same TP/PP."
            )

    @staticmethod
    def _read_tensor_index(rank_path: Path) -> dict[str, Any]:
        with (rank_path / _TENSOR_INDEX_NAME).open(
            "r", encoding="utf-8"
        ) as file:
            index = json.load(file)
        if not isinstance(index, dict) or not index:
            raise RuntimeError(
                f"Invalid or empty tensor index in {rank_path}."
            )
        return index

    def _validate_checkpoint_metadata(
        self,
        expected_state: Mapping[str, torch.Tensor],
        tensor_index: Mapping[str, Any],
        rank_path: Path,
    ) -> None:
        expected_names = set(expected_state)
        checkpoint_names = set(tensor_index)

        missing = expected_names - checkpoint_names
        unexpected = checkpoint_names - expected_names
        if missing or unexpected:
            raise RuntimeError(
                "ServerlessLLM state_dict does not match the current vLLM "
                f"runtime model. missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}. The checkpoint must be "
                "saved with the same vLLM model implementation and TP/PP."
            )

        # Bounds/schema validation is mandatory, including with strict=False.
        self._validate_tensor_index_bounds(tensor_index, rank_path)

        if not self.strict:
            return

        metadata_errors: list[str] = []
        for name, expected in expected_state.items():
            entry = tensor_index[name]
            if not isinstance(entry, (list, tuple)) or len(entry) < 5:
                metadata_errors.append(f"{name}: invalid tensor-index entry")
                continue

            _, _, saved_shape, saved_stride, saved_dtype = entry[:5]
            expected_shape = tuple(expected.shape)
            expected_stride = tuple(expected.stride())
            expected_dtype = str(expected.dtype)

            if tuple(saved_shape) != expected_shape:
                metadata_errors.append(
                    f"{name}: shape {tuple(saved_shape)} != {expected_shape}"
                )
            if tuple(saved_stride) != expected_stride:
                metadata_errors.append(
                    f"{name}: stride {tuple(saved_stride)} != "
                    f"{expected_stride}"
                )
            if saved_dtype != expected_dtype:
                metadata_errors.append(
                    f"{name}: dtype {saved_dtype} != {expected_dtype}"
                )
            if not expected.is_contiguous():
                metadata_errors.append(
                    f"{name}: non-contiguous runtime state is unsupported"
                )

        if metadata_errors:
            preview = "; ".join(metadata_errors[:20])
            suffix = "" if len(metadata_errors) <= 20 else " ..."
            raise RuntimeError(
                "ServerlessLLM tensor metadata mismatch: "
                f"{preview}{suffix}"
            )

    @staticmethod
    def _validate_tensor_index_bounds(
        tensor_index: Mapping[str, Any], rank_path: Path,
    ) -> None:
        """Validate byte ranges before passing any metadata to native code."""
        single = rank_path / "tensor.data"
        shards: dict[int, Path] = {}
        for path in rank_path.glob("tensor.data_*"):
            suffix = path.name.removeprefix("tensor.data_")
            if not suffix.isascii() or not suffix.isdigit():
                continue
            number = int(suffix)
            if number in shards:
                raise RuntimeError(f"Duplicate tensor data shard number: {path}")
            shards[number] = path
        if single.exists() or single.is_symlink():
            if shards:
                raise RuntimeError(f"Ambiguous tensor data files in {rank_path}")
            data_paths = [single]
        else:
            if not shards or sorted(shards) != list(range(len(shards))):
                raise RuntimeError(
                    f"Missing/nonconsecutive tensor data shards in {rank_path}"
                )
            data_paths = [shards[number] for number in range(len(shards))]

        data_size = 0
        for path in data_paths:
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"Tensor data must be a regular file: {path}")
            data_size += file_stat.st_size
        if not 0 < data_size <= _MAX_NATIVE_SIZE:
            raise RuntimeError(f"Invalid tensor data size in {rank_path}: {data_size}")

        allocation_size = 0
        seen_ranges: set[tuple[int, int]] = set()
        for name, entry in tensor_index.items():
            def fail(reason: str) -> None:
                raise RuntimeError(f"Invalid tensor index for {name!r}: {reason}")

            if not isinstance(name, str) or not name:
                fail("tensor name must be a nonempty string")
            if not isinstance(entry, (list, tuple)) or len(entry) != 5:
                fail("expected [offset, size, shape, stride, dtype]")
            offset, size, shape, stride, dtype = entry
            if type(offset) is not int or not 0 <= offset <= _MAX_NATIVE_SIZE:
                fail("offset must be a nonnegative native-range integer")
            if type(size) is not int or not 0 < size <= _MAX_NATIVE_SIZE:
                fail("size must be a positive native-range integer")
            if size > data_size or offset > data_size - size:
                fail("byte range exceeds the checkpoint data")
            if (
                not isinstance(shape, (list, tuple))
                or not isinstance(stride, (list, tuple))
                or len(shape) != len(stride)
                or any(type(v) is not int or not 0 <= v <= _MAX_NATIVE_SIZE
                       for v in (*shape, *stride))
            ):
                fail("shape/stride require equal lengths and nonnegative integers")
            if not isinstance(dtype, str) or dtype not in _DTYPE_ELEMENT_SIZES:
                fail("unsupported native dtype")
            numel = 1
            for dimension in shape:
                numel *= dimension
                if numel > _MAX_NATIVE_SIZE:
                    fail("tensor element count exceeds the native integer range")
            elements = 0 if 0 in shape else 1 + sum(
                (dimension - 1) * step for dimension, step in zip(shape, stride)
            )
            required_size = elements * _DTYPE_ELEMENT_SIZES[dtype]
            if required_size > size:
                fail("declared size is smaller than the tensor storage span")
            # load_dict packs unique (offset, size) records into one allocation.
            if (offset, size) not in seen_ranges:
                seen_ranges.add((offset, size))
                allocation_size += size
                if allocation_size > _MAX_NATIVE_SIZE:
                    fail("packed allocation exceeds the native integer range")

    @staticmethod
    def _release_initialized_state(
        state_vars: Mapping[str, torch.Tensor],
        target_device: torch.device,
    ) -> None:
        # Clear every distinct Parameter/Buffer object, including tied aliases,
        # so PyTorch's initialization allocation can be returned before the
        # ServerlessLLM backing allocation is created.
        seen: set[int] = set()
        for value in state_vars.values():
            object_id = id(value)
            if object_id in seen:
                continue
            seen.add(object_id)
            value.data = torch.empty(
                0,
                dtype=value.dtype,
                device=target_device,
            )

    @staticmethod
    def _bind_restored_state(
        state_vars: Mapping[str, torch.Tensor],
        loaded_state: Mapping[str, torch.Tensor],
        aliases: Mapping[str, _AliasSpec],
        target_device: torch.device,
    ) -> None:
        target_index = target_device.index
        if target_index is None:
            target_index = torch.cuda.current_device()

        for name, tensor in loaded_state.items():
            if tensor.device.type != "cuda":
                raise RuntimeError(
                    f"Restored tensor {name!r} is on {tensor.device}, "
                    f"expected {target_device}."
                )
            if tensor.device.index not in (None, target_index):
                raise RuntimeError(
                    f"Restored tensor {name!r} is on device "
                    f"{tensor.device.index}, expected {target_index}."
                )
            state_vars[name].data = tensor

        for alias_name, spec in aliases.items():
            owner = loaded_state[spec.owner_name]
            alias = owner.as_strided(
                size=spec.size,
                stride=spec.stride,
                storage_offset=(
                    owner.storage_offset()
                    + spec.relative_storage_offset
                ),
            )
            state_vars[alias_name].data = alias

    @staticmethod
    @torch.inference_mode()
    def _clone_final_loaded_state(
        loaded_state: Mapping[str, torch.Tensor],
        target_device: torch.device,
        layout: _ParallelLayout,
    ) -> dict[str, torch.Tensor]:
        """Clone owner tensors once more immediately before final binding.

        The standalone diagnostic script already has an optional clone inside
        ``sllm_store.torch.load_dict``. Repeating the clone here is deliberate:
        it proves that the tensors ultimately assigned to the vLLM model were
        allocated by PyTorch on the current rank after Store loading completed.
        """
        cloned_state: dict[str, torch.Tensor] = {}
        total_bytes = 0

        for name, tensor in loaded_state.items():
            if tensor.device.type != "cuda":
                raise RuntimeError(
                    f"Cannot clone restored tensor {name!r} on {tensor.device}; "
                    f"expected {target_device}."
                )
            cloned = tensor.detach().clone(memory_format=torch.preserve_format)
            cloned_state[name] = cloned
            total_bytes += cloned.numel() * cloned.element_size()

        torch.cuda.synchronize(target_device)
        logger.warning(
            "[SLLM_FINAL_REBIND] pid=%d tp_rank=%d/%d tensors=%d "
            "cloned_gib=%.3f device=%s",
            os.getpid(),
            layout.tp_rank,
            layout.tp_size,
            len(cloned_state),
            total_bytes / 2**30,
            target_device,
        )
        return cloned_state

    @staticmethod
    @torch.inference_mode()
    def _probe_final_model_parameter(
        state_vars: Mapping[str, torch.Tensor],
        loaded_state: Mapping[str, torch.Tensor],
        target_device: torch.device,
        layout: _ParallelLayout,
    ) -> None:
        """Probe the tensors actually bound to the final vLLM model.

        This runs after ``_bind_restored_state`` and compares one decoder
        linear weight with a fresh local-HBM clone using the same
        ``torch.nn.functional.linear`` operation.
        """
        if os.environ.get("SLLM_RUNTIME_FORWARD_PROBE", "0") != "1":
            return

        candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, loaded in loaded_state.items():
            bound = state_vars.get(name)
            if bound is None:
                continue
            if (
                bound.device.type == "cuda"
                and bound.ndim == 2
                and bound.numel() > 0
                and bound.is_floating_point()
                and bound.is_contiguous()
            ):
                candidates.append((name, bound, loaded))

        if not candidates:
            logger.warning(
                "[SLLM_FINAL_PARAM_GEMM] pid=%d tp_rank=%d/%d "
                "no suitable 2D CUDA parameter found",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
            )
            return

        decoder_tokens = (
            "qkv_proj.weight",
            "query_key_value.weight",
            "fc1.weight",
            "fc2.weight",
            "gate_up_proj.weight",
            "down_proj.weight",
        )
        decoder_candidates = [
            item
            for item in candidates
            if any(token in item[0] for token in decoder_tokens)
        ]
        name, bound, loaded = max(
            decoder_candidates or candidates,
            key=lambda item: item[1].numel() * item[1].element_size(),
        )

        torch.cuda.set_device(target_device)
        local_weight = bound.detach().clone(
            memory_format=torch.contiguous_format
        )
        probe_input = torch.ones(
            (1, bound.shape[1]),
            dtype=bound.dtype,
            device=target_device,
        )

        logger.warning(
            "[SLLM_FINAL_PARAM_META] pid=%d tp_rank=%d/%d name=%s "
            "device=%s dtype=%s shape=%s stride=%s contiguous=%s "
            "storage_offset=%d bound_ptr=0x%x loaded_ptr=0x%x "
            "same_loaded_ptr=%s alignment256=%d",
            os.getpid(),
            layout.tp_rank,
            layout.tp_size,
            name,
            bound.device,
            bound.dtype,
            tuple(bound.shape),
            tuple(bound.stride()),
            bound.is_contiguous(),
            bound.storage_offset(),
            bound.data_ptr(),
            loaded.data_ptr(),
            bound.data_ptr() == loaded.data_ptr(),
            bound.data_ptr() % 256,
        )

        # Warm both paths before measuring, so one-time BLAS/JIT setup is not
        # attributed to either the bound or cloned weight.
        bound_output = torch.nn.functional.linear(probe_input, bound)
        local_output = torch.nn.functional.linear(probe_input, local_weight)
        torch.cuda.synchronize(target_device)

        start = time.perf_counter()
        bound_output = torch.nn.functional.linear(probe_input, bound)
        torch.cuda.synchronize(target_device)
        bound_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        local_output = torch.nn.functional.linear(probe_input, local_weight)
        torch.cuda.synchronize(target_device)
        local_ms = (time.perf_counter() - start) * 1000.0

        logger.warning(
            "[SLLM_FINAL_PARAM_GEMM] pid=%d tp_rank=%d/%d name=%s "
            "bound_ms=%.3f local_clone_ms=%.3f ratio=%.2fx "
            "output_shape=%s",
            os.getpid(),
            layout.tp_rank,
            layout.tp_size,
            name,
            bound_ms,
            local_ms,
            bound_ms / max(local_ms, 1e-9),
            tuple(bound_output.shape),
        )

        del bound_output
        del local_output
        del probe_input
        del local_weight
        torch.cuda.empty_cache()

    @staticmethod
    @torch.inference_mode()
    def _probe_tp_all_reduce(
        target_device: torch.device,
        layout: _ParallelLayout,
    ) -> None:
        """Benchmark the exact vLLM tensor-parallel all-reduce path.

        The first arriving rank can wait for the other rank during warmup; that
        wait is intentionally excluded from the timed section.  Two payloads
        approximate one-token decode and the short prompt used by the probe.
        """
        if os.environ.get("SLLM_RUNTIME_FORWARD_PROBE", "0") != "1":
            return

        if layout.tp_size <= 1:
            logger.warning(
                "[SLLM_TP_ALLREDUCE] pid=%d tp_rank=%d/%d skipped=TP1",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
            )
            return

        warmup_loops = max(
            1,
            int(os.environ.get("SLLM_TP_AR_WARMUP_LOOPS", "4")),
        )
        timed_loops = max(
            1,
            int(os.environ.get("SLLM_TP_AR_LOOPS", "20")),
        )
        tp_group = get_tp_group()
        torch.cuda.set_device(target_device)

        def run_case(
            path: str,
            label: str,
            shape: tuple[int, ...],
            operation,
        ) -> None:
            probe = torch.zeros(
                shape,
                dtype=torch.float16,
                device=target_device,
            )
            for _ in range(warmup_loops):
                operation(probe)
            torch.cuda.synchronize(target_device)

            start = time.perf_counter()
            for _ in range(timed_loops):
                operation(probe)
            torch.cuda.synchronize(target_device)
            elapsed = time.perf_counter() - start

            logger.warning(
                "[SLLM_TP_ALLREDUCE] pid=%d tp_rank=%d/%d path=%s "
                "case=%s shape=%s bytes=%d loops=%d total_ms=%.3f "
                "avg_ms=%.3f",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
                path,
                label,
                shape,
                probe.numel() * probe.element_size(),
                timed_loops,
                elapsed * 1000.0,
                elapsed * 1000.0 / timed_loops,
            )
            del probe

        cases = (
            ("decode_hidden", (2048,)),
            ("prompt5_hidden", (5, 2048)),
        )

        try:
            for label, shape in cases:
                run_case(
                    "vllm_group",
                    label,
                    shape,
                    tp_group.all_reduce,
                )
        except Exception as exc:
            logger.exception(
                "[SLLM_TP_ALLREDUCE_ERROR] pid=%d tp_rank=%d/%d "
                "path=vllm_group error=%s: %s",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
                type(exc).__name__,
                exc,
            )

        # Compare with torch.distributed on the same TP process group.  A large
        # gap between these two paths points at vLLM's communicator wrapper;
        # two equally slow paths point lower, at RCCL/DTK/virtualization.
        try:
            import torch.distributed as dist

            device_group = tp_group.device_group

            def dist_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
                dist.all_reduce(tensor, group=device_group)
                return tensor

            for label, shape in cases:
                run_case(
                    "torch_dist",
                    label,
                    shape,
                    dist_all_reduce,
                )
        except Exception as exc:
            logger.exception(
                "[SLLM_TP_ALLREDUCE_ERROR] pid=%d tp_rank=%d/%d "
                "path=torch_dist error=%s: %s",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
                type(exc).__name__,
                exc,
            )

        torch.cuda.empty_cache()

    @staticmethod
    def _install_forward_timing_hooks(
        model: nn.Module,
        target_device: torch.device,
        layout: _ParallelLayout,
    ) -> None:
        """Time complete forwards, every OPT layer, and layer-0 children.

        Synchronizing around a module deliberately sacrifices performance for
        one diagnostic run.  It converts asynchronous device work into a clear
        per-module wall time so the roughly 10.4-second engine step can be
        assigned to decoder layers, attention, MLP, or work outside the model.
        """
        if os.environ.get("SLLM_RUNTIME_FORWARD_PROBE", "0") != "1":
            return

        max_calls = max(
            1,
            int(os.environ.get("SLLM_FORWARD_PROBE_CALLS", "12")),
        )
        named_modules = list(model.named_modules())
        layer_modules = [
            (name, module)
            for name, module in named_modules
            if module.__class__.__name__ == "OPTDecoderLayer"
        ]

        if not layer_modules:
            logger.warning(
                "[SLLM_FORWARD_PROBE_ERROR] pid=%d tp_rank=%d/%d "
                "reason=no_OPTDecoderLayer model_class=%s",
                os.getpid(),
                layout.tp_rank,
                layout.tp_size,
                model.__class__.__name__,
            )
            return

        first_layer_name = layer_modules[0][0]
        selected: list[tuple[str, str, nn.Module]] = [
            ("model", "<root>", model),
        ]
        selected.extend(
            ("layer", name, module)
            for name, module in layer_modules
        )

        first_layer_prefix = f"{first_layer_name}."
        for name, module in named_modules:
            if not name.startswith(first_layer_prefix):
                continue
            relative_name = name[len(first_layer_prefix):]
            # Direct children and grandchildren of decoder layer 0 reveal the
            # attention/MLP split without producing a full module-tree flood.
            if relative_name.count(".") <= 1:
                selected.append(("layer0_component", name, module))

        counts: dict[str, int] = collections.defaultdict(int)
        starts: dict[str, float] = {}
        input_shapes: dict[str, str] = {}
        handles = []

        def first_tensor_shape(value: Any) -> str:
            if isinstance(value, torch.Tensor):
                return str(tuple(value.shape))
            if isinstance(value, (tuple, list)):
                for child in value:
                    shape = first_tensor_shape(child)
                    if shape != "none":
                        return shape
            if isinstance(value, dict):
                for child in value.values():
                    shape = first_tensor_shape(child)
                    if shape != "none":
                        return shape
            return "none"

        def make_pre_hook(key: str):
            def pre_hook(module: nn.Module, args: tuple[Any, ...]) -> None:
                del module
                if counts[key] >= max_calls:
                    return
                torch.cuda.synchronize(target_device)
                input_shapes[key] = first_tensor_shape(args)
                starts[key] = time.perf_counter()

            return pre_hook

        def make_post_hook(kind: str, name: str, key: str):
            def post_hook(
                module: nn.Module,
                args: tuple[Any, ...],
                output: Any,
            ) -> None:
                del args, output
                if counts[key] >= max_calls or key not in starts:
                    return
                torch.cuda.synchronize(target_device)
                elapsed_ms = (
                    time.perf_counter() - starts.pop(key)
                ) * 1000.0
                counts[key] += 1
                logger.warning(
                    "[SLLM_FORWARD_TIME] pid=%d tp_rank=%d/%d kind=%s "
                    "call=%d name=%s class=%s input_shape=%s ms=%.3f",
                    os.getpid(),
                    layout.tp_rank,
                    layout.tp_size,
                    kind,
                    counts[key],
                    name,
                    module.__class__.__name__,
                    input_shapes.pop(key, "none"),
                    elapsed_ms,
                )

            return post_hook

        seen_modules: set[int] = set()
        for kind, name, module in selected:
            module_id = id(module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            key = f"{kind}:{name}"
            handles.append(module.register_forward_pre_hook(make_pre_hook(key)))
            handles.append(
                module.register_forward_hook(
                    make_post_hook(kind, name, key)
                )
            )

        # Module objects retain hook callables, but keeping handles explicitly
        # makes ownership clear and permits later removal in a live-reload API.
        model._sllm_forward_probe_handles = handles  # type: ignore[attr-defined]
        logger.warning(
            "[SLLM_FORWARD_PROBE_READY] pid=%d tp_rank=%d/%d "
            "layers=%d selected_modules=%d max_calls=%d first_layer=%s",
            os.getpid(),
            layout.tp_rank,
            layout.tp_size,
            len(layer_modules),
            len(seen_modules),
            max_calls,
            first_layer_name,
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> nn.Module:
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "This ServerlessLLM runtime-checkpoint adapter currently "
                "supports unquantized vLLM models only. Quantized models need "
                "a format that records whether weights were saved before or "
                "after quantizer post-processing."
            )

        load_device = (
            vllm_config.device_config.device
            if vllm_config.load_config.device is None
            else vllm_config.load_config.device
        )
        target_device = torch.device(load_device)
        if target_device.type != "cuda":
            raise NotImplementedError(
                "The DTK ServerlessLLM adapter requires a CUDA-like device "
                f"(Hygon DCU is exposed as cuda); got {target_device}."
            )

        # Reuse vLLM's standard lifecycle: initialize on the target device,
        # invoke this loader's load_weights(), then run all official post-load
        # processing exactly once.
        return super().load_model(vllm_config, model_config)

    def download_model(self, model_config: ModelConfig) -> None:
        # Configuration/tokenizer files must already be present locally. The
        # Store handles runtime tensor data rather than Hugging Face downloads.
        self._model_root(model_config)

    def load_weights(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        from sllm_store.torch import load_dict

        layout = self._parallel_layout()
        model_root = self._model_root(model_config)
        self._validate_manifest(model_root, layout)
        model_id, rank_path = self._model_id_for_rank(model_root, layout)

        target_device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )
        state_vars = model.state_dict(keep_vars=True)
        expected_state = self._filter_subtensors(state_vars)
        expected_names = set(expected_state)
        aliases = self._build_alias_plan(state_vars, expected_names)

        tensor_index = self._read_tensor_index(rank_path)
        self._validate_checkpoint_metadata(expected_state, tensor_index, rank_path)

        logger.info(
            "Loading ServerlessLLM runtime shard %s "
            "(tp_rank=%d/%d, pp_rank=%d/%d, device=%s)",
            model_id,
            layout.tp_rank,
            layout.tp_size,
            layout.pp_rank,
            layout.pp_size,
            target_device,
        )

        if self.release_init_weights:
            torch.cuda.synchronize(target_device)
            self._release_initialized_state(state_vars, target_device)
            del expected_state

            # Also release a previous ServerlessLLM mapping when load_weights()
            # is used as an in-place reload API. Keeping the old dictionary
            # here would require two full runtime allocations at once.
            old_backing_state = getattr(
                model, "_sllm_backing_state", None
            )
            for attribute in (
                "_sllm_backing_state",
                "_sllm_model_id",
                "_sllm_parallel_layout",
            ):
                if hasattr(model, attribute):
                    delattr(model, attribute)
            del old_backing_state

            gc.collect()
            if self.empty_cache_before_restore:
                torch.cuda.empty_cache()

        loaded_state = load_dict(
            model_id,
            {"": target_device.index},
            storage_path=os.fspath(self.storage_root),
        )

        loaded_names = set(loaded_state)
        if loaded_names != expected_names:
            missing = expected_names - loaded_names
            unexpected = loaded_names - expected_names
            raise RuntimeError(
                "ServerlessLLM restored a different state_dict than its "
                f"tensor index. missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        self._bind_restored_state(
            state_vars,
            loaded_state,
            aliases,
            target_device,
        )
        torch.cuda.synchronize(target_device)

        # The diagnostic script sets this variable when invoked with
        # --clone-weights-to-local-hbm. Rebind a second, final local-HBM copy
        # here so the A/B result cannot be explained by an intermediate
        # state_dict clone that failed to reach the model parameters.
        if os.environ.get(
            "SLLM_CLONE_WEIGHTS_TO_LOCAL_HBM", "0"
        ) == "1":
            loaded_state = self._clone_final_loaded_state(
                loaded_state,
                target_device,
                layout,
            )
            self._bind_restored_state(
                state_vars,
                loaded_state,
                aliases,
                target_device,
            )
            torch.cuda.synchronize(target_device)
            gc.collect()
            torch.cuda.empty_cache()

        self._probe_final_model_parameter(
            state_vars,
            loaded_state,
            target_device,
            layout,
        )

        self._probe_tp_all_reduce(
            target_device,
            layout,
        )
        self._install_forward_timing_hooks(
            model,
            target_device,
            layout,
        )

        # The returned tensors own/reference the external backing allocation.
        # Retaining them prevents the allocation from being released while the
        # vLLM model is still serving requests.
        model._sllm_backing_state = loaded_state  # type: ignore[attr-defined]
        model._sllm_model_id = model_id  # type: ignore[attr-defined]
        model._sllm_parallel_layout = layout  # type: ignore[attr-defined]

        for name, parameter in model.named_parameters(recurse=True):
            if parameter.device.type != "cuda":
                raise RuntimeError(
                    f"Parameter {name!r} was not restored to the DCU: "
                    f"device={parameter.device}."
                )
        for name, buffer in model.named_buffers(recurse=True):
            if buffer.device.type != "cuda":
                raise RuntimeError(
                    f"Buffer {name!r} is on {buffer.device}; expected DCU."
                )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def save_model(model: nn.Module, output_path: str) -> str:
        from sllm_store.torch import save_dict

        layout = ServerlessLLMLoader._parallel_layout()
        output_root = Path(output_path).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        rank_path = output_root / layout.rank_dir_name
        rank_path.mkdir(parents=True, exist_ok=True)

        state_dict = ServerlessLLMLoader._filter_subtensors(
            model.state_dict()
        )
        unsupported = [
            name for name, tensor in state_dict.items()
            if not tensor.is_contiguous()
        ]
        if unsupported:
            raise RuntimeError(
                "Cannot save non-contiguous runtime state tensors: "
                f"{unsupported[:20]}"
            )

        cpu_state_dict = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in state_dict.items()
        }
        save_dict(cpu_state_dict, os.fspath(rank_path))

        ServerlessLLMLoader._atomic_write_json(
            rank_path / _RANK_META_NAME,
            {
                "format_version": _FORMAT_VERSION,
                "model_parallel_rank": layout.model_parallel_rank,
                "tensor_parallel_rank": layout.tp_rank,
                "tensor_parallel_size": layout.tp_size,
                "pipeline_parallel_rank": layout.pp_rank,
                "pipeline_parallel_size": layout.pp_size,
                "tensor_count": len(cpu_state_dict),
            },
        )

        try:
            import vllm
            vllm_version = getattr(vllm, "__version__", "unknown")
        except Exception:
            vllm_version = "unknown"

        ServerlessLLMLoader._atomic_write_json(
            output_root / _MANIFEST_NAME,
            {
                "format": "serverless_llm_vllm_runtime_state",
                "format_version": _FORMAT_VERSION,
                "rank_layout": "pp_major_tp_minor",
                "rank_count": layout.model_parallel_size,
                "tensor_parallel_size": layout.tp_size,
                "pipeline_parallel_size": layout.pp_size,
                "model_class": (
                    f"{model.__class__.__module__}."
                    f"{model.__class__.__qualname__}"
                ),
                "torch_version": torch.__version__,
                "vllm_version": vllm_version,
                "quantized_runtime_state_supported": False,
            },
        )

        return os.fspath(rank_path)


def save_serverless_llm_state(model: nn.Module, output_path: str) -> str:
    """Picklable entry point used by ``LLMEngine.apply_model``."""
    return ServerlessLLMLoader.save_model(model, output_path)


