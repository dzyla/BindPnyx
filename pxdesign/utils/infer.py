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

import hashlib
import json
import logging
import os
import sys
import urllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from ml_collections.config_dict import ConfigDict
from protenix.config import parse_configs
from protenix.data.parser import DistillationMMCIFParser
from protenix.utils.file_io import dump_gzip_pickle, load_gzip_pickle
from pxdbench.pxd_configs.eval import eval_configs

from pxdesign.configs.configs_base import configs as configs_base
from pxdesign.configs.configs_data import data_configs
from pxdesign.configs.configs_infer import inference_configs
from pxdesign.data.utils import pdb_to_cif

URL = {
    "pxdesign_v0.1.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt",
    "protenix_base_default_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
    "protenix_mini_default_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
    "protenix_mini_tmpl_v0.5.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/protenix_mini_tmpl_v0.5.0.pt",
    "ccd_components_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif",
    "ccd_components_rdkit_mol_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl",
    "pdb_cluster_file": "https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt",
}

ALIASES = {
    "N_sample": "sample_diffusion.N_sample",
    "N_step": "sample_diffusion.N_step",
    "eta_type": "sample_diffusion.eta_schedule.type",
    "eta_min": "sample_diffusion.eta_schedule.min",
    "eta_max": "sample_diffusion.eta_schedule.max",
    "gamma0": "sample_diffusion.gamma0",
    "gamma_min": "sample_diffusion.gamma_min",
    "sample_diffusion_chunk_size": "infer_setting.sample_diffusion_chunk_size",
}

logger = logging.getLogger(__name__)


def download_inference_cache(configs) -> None:
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)

        status = f"\r[{bar}] {percent:.1f}%"
        print(status, end="", flush=True)

        if downloaded >= total_size:
            print()

    def download_from_url(tos_url, checkpoint_path, check_weight=True):
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )
        if check_weight:
            try:
                ckpt = torch.load(checkpoint_path)
                del ckpt
            except:
                os.remove(checkpoint_path)
                raise RuntimeError(
                    "Download model checkpoint failed, please download by yourself with "
                    f"wget {tos_url} -O {checkpoint_path}"
                )

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not os.path.exists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}... to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    checkpoint_path = os.path.join(
        configs.load_checkpoint_dir, f"{configs.model_name}.pt"
    )
    if not os.path.exists(checkpoint_path):
        os.makedirs(configs.load_checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}... to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    # Download protenix checkpoints ONLY when a Protenix filter will run.
    # This loop used to be unconditional, which made a Boltz-only run fail
    # before it started: a broken checkpoint symlink reads as missing, so it
    # tried to fetch through the dead link and raised FileNotFoundError on the
    # symlink's absent target. Protenix weights are ~500 MB each and are not
    # needed at all when eval_protenix / eval_protenix_mini are both off.
    binder_cfg = getattr(getattr(configs, "eval", None), "binder", None)
    needs_protenix = binder_cfg is None or any(
        bool(getattr(binder_cfg, attr, False))
        for attr in ("eval_protenix", "eval_protenix_mini")
    )
    if needs_protenix:
        for model_name in [
            "protenix_base_default_v0.5.0",
            "protenix_mini_default_v0.5.0",
            "protenix_mini_tmpl_v0.5.0",
        ]:
            checkpoint_path = os.path.join(
                configs.load_checkpoint_dir, f"{model_name}.pt"
            )
            if not os.path.exists(checkpoint_path):
                tos_url = URL[model_name]
                logger.info(
                    f"Downloading model checkpoint from\n {tos_url}... to "
                    f"{checkpoint_path}"
                )
                download_from_url(tos_url, checkpoint_path)
    else:
        logger.info(
            "Skipping Protenix checkpoint download: eval_protenix and "
            "eval_protenix_mini are both disabled."
        )

    # set checkpoint dir for ptx tools in PXDesignBench
    if hasattr(configs, "eval"):
        configs.eval.binder.tools.ptx.load_checkpoint_dir = configs.load_checkpoint_dir
        configs.eval.binder.tools.ptx_mini.load_checkpoint_dir = (
            configs.load_checkpoint_dir
        )


def remap_arg_key(key: str) -> str:
    if key.startswith("--"):
        name = key[2:]
        mapped = ALIASES.get(name, name)
        return "--" + mapped
    return key


def parse_sys_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    remapped = []

    i = 0
    while i < len(argv):
        k = argv[i]
        # if k starts with "--", check whether it matches alias
        if k.startswith("--") and i + 1 < len(argv):
            remapped.append(remap_arg_key(k))
            remapped.append(argv[i + 1])
            i += 2
        else:
            remapped.append(k)
            i += 1

    return " ".join(remapped)


def get_configs(argv=None) -> ConfigDict:
    configs = {
        **configs_base,
        **{"data": data_configs},
        **inference_configs,
        **{"eval": eval_configs},
    }
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(argv),
        fill_required_with_null=True,
    )
    return configs


class DisableLogging:
    def __enter__(self):
        logging.disable(logging.WARNING)

    def __exit__(self, exc_type, exc, tb):
        logging.disable(logging.NOTSET)


# -------------------------
# Residue numbering helpers
# -------------------------
# Moved to pxdesign/utils/residue_numbering.py so they can be imported without
# torch/protenix/biotite, which the targeting-contract tests require. Re-exported
# here so every existing caller and test keeps working; there is one
# implementation, not two.
from pxdesign.utils.residue_numbering import (  # noqa: E402,F401
    ResidueMaps,
    build_chain_mapping,
    build_residue_maps,
    convert_crop_auth_to_new,
    convert_hotspot_auth_to_new,
    format_ranges,
    parse_ranges,
    record_resolved_hotspots,
    rewrite_input_dict_inplace,
    validate_hotspots,
    _structure_residue_index,
)

# -------------------------
# Main entry: convert_to_bioassembly_dict
# -------------------------


def convert_to_bioassembly_dict(input_dict: dict, out_dir: str | None = None):
    """
    Returns:
      - if input is already .pkl.gz: (str_file)  (kept as your original behavior)
      - else: (out_path, chain_mapping)
    """
    assert "condition" in input_dict, "input_dict must have 'condition' key"
    cond_dict = input_dict["condition"]
    str_file = cond_dict["structure_file"]
    if out_dir is None:
        out_dir = os.path.dirname(str_file)

    if str_file.endswith(".pkl.gz"):
        # Already in the model's own numbering, so nothing to remap - but the
        # hotspots were never checked against it either. Validate and record.
        if input_dict.get("hotspot"):
            try:
                prebuilt = load_gzip_pickle(str_file)
                aa = prebuilt["atom_array"] if isinstance(prebuilt, dict) else prebuilt
                validate_hotspots(input_dict["hotspot"], aa)
            except (OSError, KeyError, TypeError, AttributeError) as exc:
                if isinstance(exc, KeyError) and "hotspot residues not found" in str(exc):
                    raise
                logger.warning(f"could not validate hotspots against {str_file}: {exc}")
        record_resolved_hotspots(input_dict, out_dir)
        return str_file

    chain_mapping: dict[str, str] = {}
    residue_maps: ResidueMaps | None = None
    atom_array = None

    if str_file.endswith(".cif"):
        parser = DistillationMMCIFParser(str_file)
        d = parser.get_structure_dict()
        # This branch previously built no chain mapping and never called
        # rewrite_input_dict_inplace, so a CIF's auth numbering was passed
        # through unconverted while the .pdb branch converted it - the same
        # input in two formats meant two different things.
        atom_array = d.get("atom_array") if isinstance(d, dict) else None
        if atom_array is not None and input_dict.get("hotspot"):
            try:
                residue_maps = build_residue_maps(atom_array)
                input_dict["hotspot"] = convert_hotspot_auth_to_new(
                    input_dict["hotspot"], residue_maps
                )
            except (AttributeError, KeyError) as exc:
                logger.warning(
                    f"CIF hotspot remap skipped ({exc}); validating as-is instead"
                )
            validate_hotspots(input_dict["hotspot"], atom_array)

    elif str_file.endswith(".pdb"):
        cif_file = os.path.join(out_dir, os.path.basename(str_file)[:-4] + ".cif")
        atom_array = pdb_to_cif(str_file, cif_file)

        filter_chains = cond_dict.get("filter", {}).get("chain_id", [])
        chain_mapping = build_chain_mapping(
            atom_array.auth_asym_id,
            atom_array.chain_id,
            keep_chains=filter_chains if filter_chains else None,
        )
        residue_maps = build_residue_maps(atom_array)

        rewrite_input_dict_inplace(
            input_dict,
            chain_mapping=chain_mapping,
            residue_maps=residue_maps,
        )
        if input_dict.get("hotspot"):
            validate_hotspots(input_dict["hotspot"], atom_array)
        parser = DistillationMMCIFParser(cif_file)
        d = parser.get_structure_dict()

    else:
        raise ValueError(f"Unsupported structure file! {str_file}")

    out_path = Path(out_dir) / f"{Path(str_file).stem}.pkl.gz"
    assert str(out_path).endswith(".pkl.gz"), "Bioassembly dict should end with .pkl.gz"
    dump_gzip_pickle(d, out_path)
    input_dict["condition"]["structure_file"] = str(out_path)
    record_resolved_hotspots(input_dict, out_dir)

    return d


def configure_runtime_env(
    use_fast_ln: bool = False, use_deepspeed_evo: bool = False
) -> None:
    """
    Independent runtime knobs:
      - use_fast_ln -> LAYERNORM_TYPE
      - use_deepspeed_evo  -> DEEPSPEED_EVO (+ CUTLASS dependency)
    """

    # LayerNorm
    if use_fast_ln:
        os.environ["LAYERNORM_TYPE"] = "fast_layernorm"

    # DeepSpeed Evo: fully independent
    os.environ["DEEPSPEED_EVO"] = "true" if use_deepspeed_evo else "false"

    if not use_deepspeed_evo:
        return

    if "CUTLASS_PATH" in os.environ and os.environ["CUTLASS_PATH"]:
        cutlass_path = Path(os.environ["CUTLASS_PATH"]).expanduser()
    else:
        cutlass_path = Path.home() / "cutlass"
        os.environ["CUTLASS_PATH"] = str(cutlass_path)

    if not cutlass_path.is_dir():
        print("")
        print(f"[WARNING] CUTLASS not found at: {cutlass_path}")
        print(
            "  PXDesign uses DeepSpeed Evo kernels which require NVIDIA CUTLASS v3.5.1."
        )
        print("  To install:")
        print(
            '    git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"'
        )
        print('    export CUTLASS_PATH="$HOME/cutlass"')
        print("")


def derive_seed(base_seed: int, rank: int = 0, digits: int = 6) -> int:
    mod = 10**digits
    msg = f"pxdesign|{base_seed}|{rank}".encode()
    h = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(h, "little") % mod
