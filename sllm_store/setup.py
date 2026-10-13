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
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


try:
    torch_available = True
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import ROCM_HOME
except Exception:
    torch_available = False
    CUDA_HOME = None
    ROCM_HOME = None
    print(
        "[WARNING] Unable to import torch, GPU extension build will fail. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."  # noqa: E501
    )


ROOT_DIR = os.path.dirname(__file__)


def check_nvcc_installed(cuda_home: str) -> bool:
    """Check if nvcc (NVIDIA CUDA compiler) is installed."""
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
        return True
    except Exception:
        print("[WARNING] nvcc is not installed or not found in your PATH.")
        return False


def check_hipcc_installed(rocm_home: str) -> bool:
    """Check if hipcc (AMD HIP compiler) is installed."""
    # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
    hipcc_paths = [rocm_home + "/bin/hipcc", rocm_home + "/hip/bin/hipcc"]
    for hipcc in hipcc_paths:
        try:
            _ = subprocess.check_output(
                [hipcc, "--version"], universal_newlines=True
            )
            return True
        except Exception:
            continue
    print("[WARNING] hipcc is not installed or not found in your PATH.")
    return False


def check_dtk_compiler_installed(dtk_home: str) -> bool:
    """Check whether the Hygon DTK compiler is available."""

    if not dtk_home:
        return False

    compiler_paths = [
        os.path.join(dtk_home, "bin", "hipcc"),
        os.path.join(dtk_home, "hip", "bin", "hipcc"),
        os.path.join(dtk_home, "bin", "dcc"),
    ]

    for compiler in compiler_paths:
        if not os.path.isfile(compiler):
            continue

        try:
            subprocess.check_output(
                [compiler, "--version"],
                universal_newlines=True,
                stderr=subprocess.STDOUT,
            )
            print(f"[INFO] DTK compiler found: {compiler}")
            return True
        except Exception as exc:
            print(f"[WARNING] Failed to execute {compiler}: {exc}")

    print(f"[WARNING] No usable hipcc/dcc found under {dtk_home}")
    return False


def detect_build_backend():
    """Detect which GPU backend to build based on environment variables."""

    use_dtk = os.getenv("USE_DTK", "0").lower() in ("1", "true", "yes")
    use_cuda = os.getenv("USE_CUDA", "auto").lower()
    use_rocm = os.getenv("USE_ROCM", "auto").lower()

    # DTK path priority:
    # 1. DTK_HOME
    # 2. ROCM_HOME environment variable
    # 3. Current Hygon installation
    dtk_home = (
        os.getenv("DTK_HOME")
        or os.getenv("ROCM_HOME")
        or "/opt/dtk-26.04"
    )

    # Hygon DTK support
    if use_dtk:
        if check_dtk_compiler_installed(dtk_home):
            print(f"[INFO] Hygon DTK toolkit found at: {dtk_home}")
            return "dtk", dtk_home
        raise RuntimeError(
            f"USE_DTK was requested, but no usable HIP compiler "
            f"(hipcc/dcc) was found under {dtk_home}. "
            f"Set DTK_HOME or ROCM_HOME to the DTK installation root."
        )

    # NVIDIA CUDA support
    if use_cuda != "false" and CUDA_HOME is not None:
        if check_nvcc_installed(CUDA_HOME):
            return "cuda", CUDA_HOME
        print("[WARNING] CUDA requested but nvcc not available, falling back...")

    # AMD ROCm support
    if use_rocm != "false" and ROCM_HOME is not None:
        if check_hipcc_installed(ROCM_HOME):
            return "rocm", ROCM_HOME
        print("[WARNING] ROCm available but hipcc not working, falling back...")

    raise RuntimeError(
        "No usable GPU backend found. To build for Hygon DCU set "
        "USE_DTK=1 and DTK_HOME (or ROCM_HOME) to the DTK installation root."
    )


# Detect build backend
BUILD_BACKEND, BACKEND_HOME = detect_build_backend()
print(f"[INFO] Building with backend: {BUILD_BACKEND}")


def is_ninja_available() -> bool:
    # Allow disabling ninja via environment variable
    if os.getenv("DISABLE_NINJA", "0").lower() in ("1", "true", "yes"):
        return False
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except (FileNotFoundError, OSError):
        return False
    return True


class CustomInstall(install):
    """Custom installation to ensure proto files are compiled
    and extensions are built before installation."""

    def run(self):
        self.run_command("build_ext")
        super().run()


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


install_requires = fetch_requirements("requirements.txt")

extras = {}

extras["vllm"] = [
    "vllm",
    "pillow>=10.3.0",
]

extras["test"] = [
    "pytest",
    "accelerate>=0.27.2",
    "transformers>=4.37.2",
    "parameterized",
]

sys.path.append(Path.cwd().as_posix())


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/setup.py
class cmake_build_ext(build_ext):
    did_config: Dict[str, bool] = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
        ]

        # verbose = bool(int(os.getenv('VERBOSE', '1')))
        verbose = True
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        cmake_args += [
            "-DSLLM_STORE_PYTHON_EXECUTABLE={}".format(sys.executable)
        ]

        # Add backend-specific arguments
        if BUILD_BACKEND == "dtk":
            cmake_args += [
                "-DUSE_DTK=ON",
                "-DUSE_ROCM=OFF",
                "-DUSE_CUDA=OFF",
            ]

            # Allow pointing FetchContent at local copies of glog/googletest
            # for offline DTK build machines. Only passed when the path
            # actually exists; otherwise FetchContent falls back to git.
            glog_src = os.getenv("SLLM_GLOG_SOURCE_DIR", "/opt/glog")
            if os.path.isdir(glog_src):
                cmake_args.append(
                    "-DFETCHCONTENT_SOURCE_DIR_GLOG={}".format(glog_src)
                )

            gtest_src = os.getenv(
                "SLLM_GOOGLETEST_SOURCE_DIR",
                "/root/ServerlessLLM-master/sllm_offline_deps/googletest",
            )
            if os.path.isdir(gtest_src):
                cmake_args.append(
                    "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={}".format(gtest_src)
                )

        elif BUILD_BACKEND == "cuda":
            cmake_args += [
                "-DUSE_CUDA=ON",
                "-DUSE_ROCM=OFF",
                "-DUSE_DTK=OFF",
                "-DCUDA_TOOLKIT_ROOT_DIR={}".format(BACKEND_HOME),
            ]

        elif BUILD_BACKEND == "rocm":
            cmake_args += [
                "-DUSE_ROCM=ON",
                "-DUSE_DTK=OFF",
                "-DUSE_CUDA=OFF",
                "-DHIP_ROOT_DIR={}".format(BACKEND_HOME),
            ]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(8),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            ext_target_name = remove_prefix(ext.name, "sllm_store.")
            # num_jobs = 32

            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                # str(num_jobs)
            ]

            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
            print(self.build_temp, ext_target_name)


cmdclass = {
    "build_ext": cmake_build_ext,
    "install": CustomInstall,
}

setup(
    name="serverless-llm-store",
    version="0.8.0",
    ext_modules=[
        CMakeExtension(name="sllm_store._C"),
        CMakeExtension(name="sllm_store._checkpoint_store"),
    ],
    install_requires=install_requires,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    extras_require=extras,
    entry_points={
        "console_scripts": ["sllm-store=sllm_store.cli:main"],
    },
    package_data={
        "sllm_store": ["py.typed", "*.so"],
    },
    packages=[
        "sllm_store",
        "sllm_store.proto",
    ],
    cmdclass=cmdclass,
)
