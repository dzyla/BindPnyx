# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114
from pathlib import Path

from protenix.config.extend_types import ListValue, RequiredValue

main_directory = Path(__file__).resolve().parent.parent.parent


def _empty_int_list() -> ListValue:
    """An empty int ListValue that protenix 0.5.5 can actually construct.

    protenix 0.5.5's ListValue.__init__ does `self.dtype = type(value[0])` for
    any non-None value, so `ListValue([], dtype=int)` raises IndexError at
    import and the dtype kwarg is ignored entirely. That made
    pxdesign.runner.pipeline unimportable in this env.

    The empty default is load-bearing: pipeline.py:401-404 reads
    `seeds = configs.seeds` and generates them from N_max_runs when falsy.
    ListValue(None, dtype=int) imports but is not equivalent - the config
    parser raises "config seeds not allowed to be none" for a None default
    unless --seeds is passed, turning a working default into a required flag.
    """
    value = ListValue(None, dtype=int)
    value.value = []
    return value


inference_configs = {
    "model_name": "pxdesign_v0.1.0",  # inference model selection
    "seeds": _empty_int_list(),
    "dump_dir": "./output",
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": str(main_directory / "release_data/checkpoint"),
    "num_workers": 16,
    "use_msa": True,
    "use_fast_ln": True,
}
