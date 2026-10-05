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

"""
Unified diffusion-only design pipeline.

Presets:
- preview  : AF2-only pipeline with lighter sampling and ranking.
- extended : AF2 + Protenix combined ranking.

There are two distinct ranking / evaluation logics:
  - "preview"  : AF2-only ranking rules
  - "extended" : joint AF2 + Protenix ranking rules
The ranking logic used is completely decoupled from the CLI:
it is automatically inferred from the actual outputs instead of any user flag.

Decision rule:
  - If PTX-related columns exist in the final summary table
      (ptx_* or ptx_mini_*)
      → use the "extended" ranking pipeline
  - Otherwise
      → fall back to the "preview" (AF2-only) ranking pipeline

Likewise, whether target-template logic (use_target_template_or_not) is executed
is automatically determined by whether the Protenix filter is enabled in the
evaluation configuration, instead of being controlled by CLI arguments.

Assumes a SINGLE task (i.e., no per-task grouping loops).
"""

import argparse
import copy
import json
import logging
import os
import time

import torch
from protenix.config import save_config
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from pxdbench.run import run_task
from pxdbench.utils import convert_cifs_to_pdbs, str2bool

from pxdesign.runner.dumper import DataDumper
from pxdesign.runner.helpers import save_top_designs, use_target_template_or_not
from pxdesign.runner.inference import InferenceRunner
from pxdesign.runner.presets import PRESETS
from pxdesign.utils.infer import (
    convert_to_bioassembly_dict,
    derive_seed,
    download_inference_cache,
    get_configs,
)
from pxdesign.utils.inputs import process_input_file
from pxdesign.utils.pipeline import check_tool_weights
from pxdbench.targets.preparation import (
    apply_policy_dict_to_configs,
    convert_and_resolve,
)
from pxdbench.targets.registry import Registry as _Registry
from pxdbench.targets.registry import early_stop_key as _early_stop_key

_target_registry = _Registry.load_default()

logger = logging.getLogger(__name__)


def _get_overridden_keys(argv) -> set:
    """
    Inspect raw argv and infer which long-form options were explicitly set
    by the user, so that presets do NOT overwrite those values.
    """
    if argv is None:
        return set()

    overridden = set()
    it = iter(argv)
    for token in it:
        if not token.startswith("-"):
            continue
        if token.startswith("--"):
            # --foo or --foo=bar
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overridden.add(name.replace("-", "_"))
    return overridden


class DesignPipeline(InferenceRunner):
    def __init__(
        self,
        *args,
        use_ptx_filter: bool = False,
        use_boltz_filter: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_run = 0
        self.use_ptx_filter = use_ptx_filter
        self.use_boltz_filter = use_boltz_filter

    def resolve_use_target_template(self, pipeline_args, orig_seqs, seed):
        """Decide the Protenix target-template question.

        Returns False immediately under the Boltz scorer: the heuristic runs a
        target-only PROTENIX prediction and mutates
        configs.eval.binder.tools.ptx, and the Boltz calibration is
        template-free, so there is nothing to decide and nothing to load.
        """
        if self.use_boltz_filter or not self.use_ptx_filter:
            return False

        use_target_template = None
        if DIST_WRAPPER.rank == 0:
            task_name = list(orig_seqs.keys())[0]
            gt_cif_path = os.path.join(
                self.dump_dir,
                task_name,
                f"seed_{seed}",
                "predictions",
                f"{task_name}_sample_0.cif",
            )
            use_target_template = use_target_template_or_not(
                self.configs.eval,
                pipeline_args,
                gt_cif_path,
                orig_seqs[task_name],
                task_name,
                os.path.join(self.configs.dump_dir, "target_pred"),
                device="cuda:0",
                seed=seed,
            )
        use_templ_list = DIST_WRAPPER.all_gather_object(use_target_template)
        print("use_templ_list: ", use_templ_list)
        use_target_template = [x for x in use_templ_list if x is not None][0]
        print("use_target_template: ", use_target_template)

        # PATCH (v2 compat): no template-aware Protenix-v2 checkpoint exists
        # (only protenix_mini_tmpl_v0.5.0 supports use_template=True). Before
        # this fix, use_target_template=True with model_name=="protenix-v2"
        # silently dropped the target chain during patch_with_orig_seqs,
        # producing single-chain (binder-only) Protenix inputs and a
        # structurally-undefined ptx_iptm/ptx_iptm_binder=0.0 for every
        # design. Force full-complex co-folding (no templating) for v2.
        if (
            use_target_template
            and self.configs.eval.binder.tools.ptx.model_name == "protenix-v2"
        ):
            print(
                "[WARN] use_target_template heuristic requested template-mode, "
                "but no template-aware Protenix-v2 checkpoint exists. "
                "Falling back to full-complex co-folding for Protenix-v2 "
                "(target chain will NOT be dropped)."
            )
            use_target_template = False

        if use_target_template:
            self.configs.eval.binder.tools.ptx.use_template = True
            self.configs.eval.binder.tools.ptx.use_msa = False
            if self.configs.eval.binder.tools.ptx.model_name in [
                "protenix_base_default_v0.5.0",
                "protenix_mini_default_v0.5.0",
            ]:
                self.configs.eval.binder.tools.ptx.model_name = (
                    "protenix_mini_tmpl_v0.5.0"
                )
            print("[INFO] Use target template in the Protenix filter!")
        return use_target_template

    def infer_and_eval(
        self,
        seed: int,
        run_id: int,
        pipeline_args,
        progress_per_infer: float = 10.0,
        progress_per_eval: float = 30.0,
    ):
        self.dump_dir = os.path.join(
            self.configs.dump_dir, f"global_run_{self.global_run}"
        )
        os.makedirs(self.dump_dir, exist_ok=True)
        self.dumper = DataDumper(base_dir=self.dump_dir)

        seed_everything(seed=seed, deterministic=True)
        orig_seqs = self._inference(seed)

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

        use_target_template = self.resolve_use_target_template(
            pipeline_args, orig_seqs, seed
        )

        cur_progress = (
            run_id * (progress_per_infer + progress_per_eval) + progress_per_infer
        )
        if DIST_WRAPPER.rank == 0:
            print(f"----------Current progress: {cur_progress:.2f}%----------")

        eval_inputs = []
        for task_name in orig_seqs:
            input_dir = os.path.join(
                self.dump_dir, task_name, f"seed_{seed}", "predictions"
            )
            if not os.path.exists(input_dir):
                logger.warning(f"Cannot find inference results under {input_dir}")
                continue
            pdb_dir, pdb_names, cond_chains, binder_chains = convert_cifs_to_pdbs(
                input_dir
            )
            eval_inputs.append(
                {
                    "task": "binder",
                    "name": task_name,
                    "pdb_dir": pdb_dir,
                    "pdb_names": pdb_names,
                    "cond_chains": cond_chains,
                    "binder_chains": binder_chains,
                    "out_dir": input_dir,
                    "orig_seqs": orig_seqs[task_name],
                }
            )
        results = [
            run_task(
                eval_inputs[i],
                self.configs.eval,
                device_id=DIST_WRAPPER.local_rank,
                seed=seed,
            )
            for i in range(len(eval_inputs))
        ]
        all_eval = DIST_WRAPPER.all_gather_object(results)
        all_eval = [x for sub in all_eval for x in sub]
        cur_progress = (run_id + 1) * (progress_per_infer + progress_per_eval)
        if DIST_WRAPPER.rank == 0:
            print("all eval results: ", all_eval)
            print(f"----------Current progress: {cur_progress:.2f}%----------")
        return all_eval, orig_seqs, use_target_template


# ---------- CLI & orchestration ----------


def parse_pipeline_args(argv=None):
    """
    Parse pipeline-level CLI arguments (high-level presets + core knobs).
    Remaining args are passed to get_configs (model/eval config).
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        choices=["preview", "extended", "custom"],
        default="preview",
        help=(
            "High-level pipeline preset. "
            "'preview' / 'extended' set a bundle of defaults "
            "for sampling and ranking. 'none' disables presets."
        ),
    )

    parser.add_argument(
        "--N_max_runs",
        type=int,
        default=1,
        help="Max number of global pipeline rounds.",
    )
    parser.add_argument(
        "--target_template_rmsd_thres",
        type=float,
        default=2.0,
        help="Max RMSD between GT target and prediction to treat as 'template-like'.",
    )

    # Output and ranking caps
    parser.add_argument(
        "--return_topk",
        type=int,
        default=5,
        help="How many designs to keep per task after ranking.",
    )
    parser.add_argument(
        "--min_total_return",
        type=int,
        default=10,
        help="If total successes < this, pad with failed designs up to this total.",
    )
    parser.add_argument(
        "--max_success_return",
        type=int,
        default=20,
        help="Max number of success rows to return.",
    )
    parser.add_argument(
        "--per_backbone_cap",
        type=int,
        default=1,
        help=(
            "Boltz mode only: max designs kept per backbone in the final "
            "selection. At num_seqs > 1 an uncapped selection can fill with "
            "siblings of one backbone."
        ),
    )
    parser.add_argument(
        "--extended_w_af2",
        type=float,
        default=0.5,
        help="Weight for AF2 rank in extended (AF2+Protenix) ranking.",
    )
    parser.add_argument(
        "--extended_w_ptx",
        type=float,
        default=0.5,
        help="Weight for PTX rank in extended (AF2+Protenix) ranking.",
    )

    # Early-stop knobs
    parser.add_argument(
        "--early_stop",
        type=str2bool,
        default=True,
        help="Whether to early-stop the global pipeline if enough successes are accumulated.",
    )
    parser.add_argument(
        "--allow_provisional_early_stop",
        type=str2bool,
        default=False,
        help=(
            "Allow a provisional, unvalidated gate to trigger early stopping. "
            "Off by default: with --early_stop true and "
            "--min_early_stop_successes 1, one pass of the provisional Boltz "
            "gate would otherwise end the campaign."
        ),
    )
    parser.add_argument(
        "--min_early_stop_rounds",
        type=int,
        default=0,
        help="Min number of rounds before early-stop is allowed.",
    )
    parser.add_argument(
        "--min_early_stop_successes",
        type=int,
        default=1,
        help="Min number of total successes required to trigger early-stop.",
    )

    overridden_keys = _get_overridden_keys(argv)
    pipeline_args, remaining = parser.parse_known_args(argv)

    preset_name = pipeline_args.preset
    if preset_name and preset_name != "custom":
        preset_cfg = PRESETS.get(preset_name, {})
        for key, value in preset_cfg.items():
            # Do not overwrite CLI-explicit arguments
            if key in overridden_keys:
                continue
            setattr(pipeline_args, key, value)

    return pipeline_args, remaining


def parse_args(argv=None):
    """
    Top-level argument parser:
      - parse pipeline-level knobs and presets
      - parse model/eval configs via get_configs
      - inject dtype and deepspeed flags into eval configs
    """
    pipeline_args, remaining_args = parse_pipeline_args(argv)
    configs = get_configs(remaining_args)
    for tool_name in ["ptx_mini", "ptx"]:
        configs["eval"]["binder"]["tools"][tool_name].update(
            {
                "dtype": configs.dtype,
                "use_deepspeed_evo_attention": configs.use_deepspeed_evo_attention,
            }
        )
    return configs, vars(pipeline_args)


def detect_use_ptx_filter(configs) -> bool:
    """
    Detect whether Protenix filter is enabled in eval configs.
    """
    binder_cfg = configs.eval.binder
    use_ptx = False
    for attr in ["eval_protenix", "eval_protenix_mini"]:
        if hasattr(binder_cfg, attr) and getattr(binder_cfg, attr):
            use_ptx = True
    return use_ptx


BOLTZ_EARLY_STOP_KEY = _early_stop_key(_target_registry.default_gate_id())
AF2_EARLY_STOP_KEY = "af2_easy_success.count"

#: Gates whose thresholds are not validated. A campaign must not be halted by
#: one of these without the operator saying so explicitly.
PROVISIONAL_GATE_KEYS = frozenset({BOLTZ_EARLY_STOP_KEY})


def may_early_stop(gate, allow_provisional: bool) -> bool:
    """Whether THIS gate may halt a campaign, resolved from its own metadata.

    `PROVISIONAL_GATE_KEYS` matches a fixed set of success-key STRINGS, so a
    gate identified by a registry entry rather than a hardcoded name would
    bypass the guard entirely (review A3). Ask the gate, not the list.
    """
    if not getattr(gate, "provisional", True):
        return True
    return bool(allow_provisional)


def early_stop_success_key(use_boltz_filter: bool) -> str:
    """Which summary key counts successes, resolved from the active mode."""
    return BOLTZ_EARLY_STOP_KEY if use_boltz_filter else AF2_EARLY_STOP_KEY


def resolve_early_stop(early_stop: bool, success_key: str, allow_provisional: bool):
    """Whether early stopping may run, given which gate would decide it.

    `--early_stop` defaults to True and `--min_early_stop_successes` to 1, so
    with the Boltz scorer active a SINGLE pass of the provisional gate would end
    the campaign. That gate's thresholds are inherited from a three-term gate,
    the clause that was dropped strictly loosens it, and the labelled set needed
    to validate it is not available - so it must not decide how much compute a
    campaign gets without the operator opting in.

    Returns (enabled, reason). Disabling is deliberate rather than fatal: a long
    run should not die at the finish line over a policy default.
    """
    if not early_stop:
        return False, "early_stop was not requested"
    if success_key in PROVISIONAL_GATE_KEYS and not allow_provisional:
        return False, (
            f"early stopping is DISABLED: it would be decided by {success_key!r}, "
            f"whose thresholds are provisional and unvalidated. Pass "
            f"--allow_provisional_early_stop to override, and record that you "
            f"did in the campaign notes."
        )
    return True, "enabled"


def accumulate_successes(all_eval_results, cumulative_success, success_key):
    """Accumulate per-task success counts.

    A missing key RAISES. It used to be swallowed by `.get(key, 0)`, which meant
    that with AF2 disabled the count stayed 0 forever and early stopping silently
    never fired - the flag became dead rather than failing.
    """
    cumulative_success = dict(cumulative_success or {})
    for item in all_eval_results:
        with open(item["summary_save_path"], "r") as handle:
            summary = json.load(handle)
        if success_key not in summary:
            raise KeyError(
                f"early-stop key {success_key!r} absent from "
                f"{item['summary_save_path']}. Available keys: "
                f"{sorted(summary)}. The selection policy is misconfigured; a "
                f"policy that silently counts nothing is the failure being "
                f"guarded against."
            )
        name = item["name"]
        cumulative_success[name] = cumulative_success.get(name, 0) + int(
            summary[success_key]
        )
    return cumulative_success


def detect_use_boltz_filter(configs) -> bool:
    """Whether the Boltz-2 scorer is enabled.

    Deliberately separate from detect_use_ptx_filter: that flag also gates
    resolve_use_target_template, which runs a target-only PROTENIX prediction
    and mutates configs.eval.binder.tools.ptx. Boltz must trigger neither.
    """
    binder_cfg = configs.eval.binder
    return bool(getattr(binder_cfg, "eval_boltz", False))


def main(argv=None):
    configs, p = parse_args(argv)
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = process_input_file(
        configs.input_json_path, out_dir=configs.dump_dir
    )
    download_inference_cache(configs)
    check_tool_weights(configs)

    # convert cif / pdb to bioassembly dict
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            # Prepare through the one boundary, so an `epitope` block actually
            # reaches the resolver and then the scorer. The 2026-10-01 smoke
            # run exposed what was missing here: conditioning reached the
            # diffusion while the policy never reached the backend, so
            # ep_satisfied was null for every design and a gate clause on it
            # was inert.
            prepared = convert_and_resolve(x, configs.dump_dir)
            x.clear()
            x.update(prepared)
            if x.get("resolved_policy_for_scoring"):
                apply_policy_dict_to_configs(
                    x["resolved_policy_for_scoring"], configs
                )
        configs.input_json_path = os.path.join(configs.dump_dir, "pipeline_input.json")
        with open(configs.input_json_path, "w") as f:
            json.dump(orig_inputs, f, indent=4)
        # Re-save: the config above was written BEFORE the epitope policy was
        # applied, so it recorded `epitope_policy: {}` on runs that enforced
        # one. config.yaml is what someone reads later to find out whether a
        # policy was active, and it was answering no on six runs that said yes.
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))

    if DIST_WRAPPER.world_size > 1:
        if DIST_WRAPPER.rank == 0:
            new_inputs = []
            with open(configs.input_json_path, "r") as f:
                ori_input = json.load(f)[0]
            for i in range(DIST_WRAPPER.world_size):
                new_input = copy.deepcopy(ori_input)
                new_input["name"] = ori_input["name"] + f"_chunk{i}"
                new_inputs.append(new_input)
            with open(
                os.path.join(configs.dump_dir, "chunk_pipeline_input.json"), "w"
            ) as f:
                json.dump(new_inputs, f, indent=4)
        # split N_sample over workers
        if hasattr(configs, "sample_diffusion") and hasattr(
            configs.sample_diffusion, "N_sample"
        ):
            configs.sample_diffusion.N_sample = (
                configs.sample_diffusion.N_sample // DIST_WRAPPER.world_size
            )
        configs.input_json_path = os.path.join(
            configs.dump_dir, "chunk_pipeline_input.json"
        )

    use_ptx_filter = detect_use_ptx_filter(configs)
    use_boltz_filter = detect_use_boltz_filter(configs)
    runner = DesignPipeline(
        configs,
        use_ptx_filter=use_ptx_filter,
        use_boltz_filter=use_boltz_filter,
    )

    N = p["N_max_runs"]
    seeds = configs.seeds
    if not seeds:
        base = time.time_ns()
        seeds = [(base + i) % (2**31 - 1) for i in range(N)]
    else:
        assert len(seeds) == N, "The number of seeds must equal N_max_runs"

    progress_per_infer = round(30.0 / p["N_max_runs"])
    progress_per_eval = round(60.0 / p["N_max_runs"])

    cumulative_success = {}  # name -> int

    for i in range(p["N_max_runs"]):
        with open(configs.input_json_path, "r") as f:
            cur_inputs = json.load(f)

        local_seed = derive_seed(seeds[i], DIST_WRAPPER.rank)
        runner.local_print(f"----------Pipeline with seed {local_seed}----------")
        runner.print(f"Current {len(cur_inputs)} design tasks for loop {i}:")
        runner.print(f"Current tasks: {cur_inputs}")

        all_eval_results, orig_seqs, use_target_template = runner.infer_and_eval(
            seed=local_seed,
            run_id=i,
            pipeline_args=p,
            progress_per_infer=progress_per_infer,
            progress_per_eval=progress_per_eval,
        )
        assert len(cur_inputs) == len(all_eval_results)

        # save meta info
        meta_info = {"mode": "Extended" if use_ptx_filter else "Preview"}
        if use_ptx_filter:
            # PATCH (v2 compat): same guard as infer_and_eval - no template-aware
            # Protenix-v2 checkpoint exists, so never template-condition when
            # model_name == "protenix-v2" (would silently drop the target chain).
            if use_target_template and runner.configs.eval.binder.tools.ptx.model_name == "protenix-v2":
                use_target_template = False
            if use_target_template:
                runner.configs.eval.binder.tools.ptx.use_template = True
                runner.configs.eval.binder.tools.ptx.use_msa = False
                if runner.configs.eval.binder.tools.ptx.model_name in ["protenix_base_default_v0.5.0", "protenix_mini_default_v0.5.0"]:
                    runner.configs.eval.binder.tools.ptx.model_name = (
                        "protenix_mini_tmpl_v0.5.0"
                    )
                meta_info["protenix"] = "Protenix-Mini-Templ"
            else:
                meta_info["protenix"] = "Protenix"
        if DIST_WRAPPER.rank == 0:
            task_name = cur_inputs[0]["name"]
            if DIST_WRAPPER.world_size > 1:
                task_name = task_name[: -len("_chunk*")]
            output_dir = os.path.join(configs.dump_dir, "design_outputs", task_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "task_info.json"), "w") as f:
                json.dump(meta_info, f, indent=4)

        # accumulate successes by the key the ACTIVE MODE declares; a missing
        # key raises rather than silently counting nothing
        success_key = early_stop_success_key(use_boltz_filter)
        cumulative_success = accumulate_successes(
            all_eval_results, cumulative_success, success_key
        )
        early_stop_enabled, early_stop_reason = resolve_early_stop(
            p["early_stop"], success_key, p.get("allow_provisional_early_stop", False)
        )
        if not early_stop_enabled and p["early_stop"]:
            print(f"[WARN] {early_stop_reason}")

        print(
            f"[Loop {i}] Designed: {[it['name'] for it in all_eval_results]}, "
            f"cumulative success: {cumulative_success}"
        )

        success_names = []
        for data in cur_inputs:
            task_name = data["name"]
            success_count = cumulative_success.get(task_name, 0)
            if (
                ((i + 1) >= p["min_early_stop_rounds"])
                and (success_count >= p["min_early_stop_successes"])
                and early_stop_enabled
            ) or i == p["N_max_runs"] - 1:
                success_names.append(task_name)

        runner.global_run += 1
        next_inputs = [x for x in cur_inputs if x["name"] not in success_names]
        if not next_inputs or i == p["N_max_runs"] - 1:
            print("Finish all designs!")
            if DIST_WRAPPER.rank == 0:
                save_top_designs(
                    p,
                    configs,
                    orig_seqs,
                    use_template=use_target_template,
                )
            break

    if DIST_WRAPPER.rank == 0:
        print("----------Current progress: 100.00%----------")


if __name__ == "__main__":
    main()
