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

# pylint: disable=C0114,C0301

from protenix.config.extend_types import ListValue

eval_configs = {
    "save_dir": "",
    "monomer": {
        "eval_diversity": False,
        "num_seqs": 8,
        "tools": {
            "mpnn": {
                "model_type": "ca",  # [ca, bb, soluble]
                "model_name": "v_48_020",
                "temperature": "0.1",
            }
        },
    },
    "binder": {
        "eval_diversity": False,
        "eval_binder_monomer": True,
        "eval_complex": True,
        "eval_protenix_mini": True,
        "eval_protenix": True,
        "eval_boltz": False,
        "num_seqs": 1,
        "use_gt_seq": False,
        "use_binder_seq_list": False,
        "is_cyclic": False,
        "tools": {
            "mpnn": {
                "weights": "original",  # [original, soluble]
                "rm_aa": "C",
                "temperature": "0.0001",
                "fix_interface": False,  # whether fixing interface restype or not
            },
            "af2": {
                "use_multimer": False,
                "model_ids": ListValue([0]),
                "use_initial_guess": True,
                "use_initial_atom_pos": False,
                "use_binder_template": True,
                "is_cyclic": False,
            },
            "ptx_mini": {
                "model_name": "protenix_mini_default_v0.5.0",
                "load_checkpoint_dir": "",
                "dtype": "bf16",
                "use_deepspeed_evo_attention": True,
                "N_cycle": 4,
                "N_sample": 1,
                "N_step": 2,
                "step_scale_eta": 1.0,
                "gamma0": 0,
                "use_template": False,
                "use_msa": True,
            },
            "boltz": {
                # Resolved by pxdbench.toolenv at use time: the project's own
                # .pxd/envs/boltz first, then $PXD_BOLTZ_BIN, then PATH. Never a
                # hardcoded path - an earlier revision baked one person's env
                # here and every other machine died at the first fold.
                "boltz_bin": "",
                # OVERRIDE ONLY, keyed by DESIGN chain id. Normally left empty:
                # each target entity in orig_seqs carries its own cached
                # precomputed_msa_dir, and that is the correct pairing. The
                # cached dirs are NAMED after the source structure's chains
                # (obj2_chainB, obj2_chainD) while the design's chains are A and
                # B, so a hand-written map here silently mispairs them.
                "target_msa": {},
                "a3m_name": "non_pairing.a3m",
                "binder_chain_id": "",
                "seeds_rank": 1,
                "seeds_gate": 3,
                "recycling_steps": 3,
                "diffusion_samples": 1,
                "ipdae_cutoff": 8.0,
                "ipsae_pae_cutoff": 10.0,
                # bz_ipdae only after the milestone head-to-head supports it
                "ranking_key": "bz_ipsae",
                "compare_ranking_keys": False,
                "anchor_sequence": "",
                "keep_structures": "winners",  # all | winners | none
                "timeout_s": 3600,
                "gpu": "0",
                # Concurrent `boltz predict` processes per seed. Boltz leaves
                # the GPU idle ~83% of the time (measured): the gaps are its
                # per-invocation model load and single-threaded CPU MSA
                # featurisation, so overlapping W processes fills them. Sharding
                # is by design within one seed, so results do not depend on W -
                # verified by an equivalence check. 1 = the original behaviour.
                "workers": 1,
                # Boltz's own dataloader worker count (--num_workers).
                # MEASURED, DO NOT RAISE: at 4 workers it gave no speedup
                # (0.99x) AND changed 42 of 42 metric values, max |diff| 3.88.
                # See scripts/benchmark_boltz.py and
                # docs/superpowers/specs/2026-09-30-gpu-utilisation-design.md.
                # Kept configurable only so the measurement is reproducible.
                "boltz_num_workers": 0,
                # --no_kernels disables Boltz's optimised triangular attention
                # (cuequivariance_torch). run_boltz_campaign.py passed it, so
                # the calibration ran with the kernels OFF. True preserves that.
                "boltz_no_kernels": True,
                # Verify after prediction that the complex contacts the residues
                # it was conditioned on. Hotspots are only a SOFT generation
                # condition - a per-token model feature - and nothing else ever
                # checked the result, so a design could score well at another
                # site. {"required": {"A": [25, 27]}, "forbidden": {"B": [...]},
                # "cutoff": 4.0}, in the DESIGN's residue numbering.
                "epitope_policy": {},
            },
            "ptx": {
                "model_name": "protenix-v2",
                "load_checkpoint_dir": "",
                "dtype": "bf16",
                "use_deepspeed_evo_attention": True,
                "N_cycle": 4,
                "N_sample": 1,
                "N_step": 2,
                "step_scale_eta": 1.0,
                "gamma0": 0,
                "use_template": False,
                "use_msa": True,
            },
        },
        "filters": {
            "af2_easy": {
                "pLDDT": (">", 0.8),
                "i_pTM": (">", 0.5),
                "i_pAE": ("<", 0.35),
                "bound_unbound_RMSD": ("<", 3.5),
            },
            "af2_opt": {
                "pLDDT": (">", 0.9),
                "unscaled_i_pAE": ("<", 7.0),
                "af2_binder_pred_design_rmsd": ("<", 1.5),
            },
            "ptx_mini": {
                "ptx_mini_iptm_binder": (">", 0.85),
                "ptx_mini_ptm_binder": (">", 0.88),
                "ptx_mini_pred_design_rmsd": ("<", 2.5),
            },
            "ptx": {
                "ptx_iptm_binder": (">", 0.85),
                "ptx_ptm_binder": (">", 0.88),
                "ptx_pred_design_rmsd": ("<", 2.5),
            },
            "ptx_basic": {
                "ptx_iptm_binder": (">", 0.8),
                "ptx_ptm_binder": (">", 0.8),
                "ptx_pred_design_rmsd": ("<", 2.5),
            },
            # PROVISIONAL, not calibrated: the published three-term gate's
            # interface_plddt clause has no traceable implementation, and
            # dropping it strictly LOOSENS the gate, so its true positive rate
            # is higher and its precision lower than the published figures by
            # an unmeasured amount. Never drives early stopping; it is an
            # exploratory selection rule, not a validated filter.
            "bz_gate_egfr_provisional_v1": {
                "bz_n_seeds": (">=", 3),
                "bz_pae_interface_min": ("<=", 2.0),
                "bz_ipsae": (">=", 0.5),
            },
        },
    },
}
