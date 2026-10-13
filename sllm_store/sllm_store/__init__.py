# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #


import ctypes
import os as _os


def _load_packaged_glog() -> None:
    """Preload the bundled libglog if present.

    The extensions link against a glog shipped inside the package,
    which is not on the system library search path. server.py already
    handles this for the store server; preload here so processes that
    import the package directly (e.g. vLLM EngineCore workers) work
    too. Skipped when the library is not bundled.
    """
    _glog_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "libglog.so"
    )
    if _os.path.isfile(_glog_path):
        ctypes.CDLL(_glog_path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))


_load_packaged_glog()
