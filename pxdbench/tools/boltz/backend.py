"""Boltz-2 backend for pxdbench's in-loop structure filter.

Resolution is wired through the registry:

    export PXDBENCH_BACKEND='pxdbench.tools.boltz.backend:BoltzBackend'
    export PYTHONPATH=<repo>

but INVOCATION needs BaseTask.get_boltz / boltz_predict and the eval_boltz
branch in BinderTask.run (Task 7): get_ptx passes tools.ptx or tools.ptx_mini,
never tools.boltz.

Two-tier scoring: every sequence is folded once and ranked; only the
per-backbone winner under the configured `ranking_key` is re-folded at
`seeds_gate` and gated on the 3-seed mean. Non-winners keep their honest 1-seed
values and carry `bz_n_seeds = 1`, so they fail the gate on the seed-count
clause rather than on a missing or NaN value.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pxdbench.tools.boltz import parse as bp
from pxdbench.tools.boltz import msa_check as mc
from pxdbench.tools.boltz import runner as br
from pxdbench.tools.boltz.msa_check import (
    DEFAULT_MIN_DEPTH,
    file_sha256,
    validate_target_chains,
)
from pxdbench.tools.boltz.targets import target_chains_from_orig_seqs
from pxdbench.targets.registry import Registry
from pxdbench.tools.boltz.result_schema import aggregate_policy
from pxdbench.tools.boltz.yaml_writer import (
    DEFAULT_POCKET_MAX_DISTANCE,
    validate_pocket_contacts,
    write_design_yaml,
    write_manifest,
)

# Resolved from pxdbench/targets/registry.json, not written here. A
# module-level dict plus a literal id is how one target came to be baked into
# four files; a second target then had to reuse its calibrated thresholds or
# edit source. The names are kept so existing imports keep working.
_REGISTRY = Registry.load_default()
_GATE_SPEC = _REGISTRY.gate(_REGISTRY.default_gate_id())
GATE_NAME = _GATE_SPEC.gate_id
GATE = _GATE_SPEC.thresholds

MEAN_KEYS = (
    "bz_ipdae",
    "bz_ipsae",
    "bz_pae_interface_min",
    "bz_interface_plddt",
    "bz_iptm",
    "bz_ptm",
    "bz_complex_plddt",
)

RANK_TAG = "rank"
GATE_TAG = "gate"


#: Below this, two ipSAE scores are not distinguishable from each other.
#: Measured 2026-10-02 on obj2: composition SD 0.0161, batch-size SD 0.0211,
#: between-design SD 0.0304 - a signal-to-noise of 1.88. Provisional, and
#: characterised on ONE target; re-measure before trusting it elsewhere.
#: docs/measurements/2026-10-02-oracle-discrimination.md
IPSAE_TIE_TOLERANCE = 0.03


def sample_name_of(item):
    return f"{item['name']}_seq{item['seq_idx']}"


def _jsonable(value):
    """Plain Python data, whatever config wrapper it arrived in.

    The config system hands back a ConfigDict for any nested mapping it was
    given, including `epitope_policy`. json.dump refuses it, so writing the
    calibration record killed the first campaign that ever carried a real
    policy - AFTER the GPU work, at the provenance step, with the scoring
    already done. Coerce rather than assume: this record exists to be
    readable later, and an unreadable one is the same as none.
    """
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _engagement_of(row):
    """How much of the requested patch this design actually contacts.

    The tiebreak among scores the oracle cannot tell apart. Missing or
    unparseable engagement sorts last rather than first: an unmeasured design
    does not win a tie.
    """
    try:
        value = float((row or {}).get("ep_best_patch_frac"))
    except (TypeError, ValueError):
        return -1.0
    return value if value == value else -1.0


def design_uid(item):
    """A stable identifier for one design, for the experimental loop.

    Derived from the backbone name, the sequence index and the sequence itself,
    so it survives re-running, re-scoring and moving between directories, and
    two designs with the same sequence on the same backbone collide
    deliberately. sample_name alone is only unique within one run.
    """
    payload = f"{item['name']}|{item['seq_idx']}|{item['sequence']}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def measured_boltz_version(boltz_bin):
    """The installed boltz's package version, or None when it cannot be read.

    Asks the python next to the binary (boltz lives in its own environment, so
    this interpreter cannot import it). None is honest: it says "not measured"
    rather than echoing the constant the scorer was written against.
    """
    py = os.path.join(os.path.dirname(os.path.abspath(boltz_bin)), "python")
    if not os.access(py, os.X_OK):
        return None
    try:
        res = subprocess.run(
            [py, "-c", "from importlib.metadata import version;"
                       "print(version('boltz'))"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (res.stdout or "").strip() or None


class BoltzBackend:
    """Implements the pxdbench backend protocol using Boltz-2."""

    def __init__(self, cfg=None, device=None, *args, **kwargs):
        self.cfg = dict(cfg) if cfg else {}
        self.device = device
        self.batch_id = None
        self._entries = []
        self._target_chains = []

    # -- config helpers -------------------------------------------------- #
    def _get(self, key, default=None):
        value = self.cfg.get(key, default)
        return default if value is None else value

    @property
    def ranking_key(self):
        return self._get("ranking_key", "bz_ipsae")

    @property
    def seeds_gate(self):
        return int(self._get("seeds_gate", 3))

    @property
    def seeds_rank(self):
        return int(self._get("seeds_rank", 1))

    def _pocket_contacts(self):
        """[(chain, res), ...] to constrain the binder against, or [].

        Opt-in. Boltz is otherwise template-free and re-docks the binder
        wherever it likes: measured across 7 designs in 3 runs, the diffusion
        engaged 3.0-3.5 of 6 requested hotspots and the scored pose 1.0-2.5,
        the two poses agreeing on the engaged set once. Constraining the pocket
        changes the Boltz input, so GATE no longer applies - see
        `gate_applies` in the calibration record.
        """
        return [
            (str(chain), int(res))
            for chain, res in (self._get("pocket_contacts", []) or [])
        ]

    # -- protocol -------------------------------------------------------- #
    def prepare_json(
        self,
        pdb_dir,
        data_list,
        dump_dir="",
        binder_chain_idx=None,
        orig_seqs=None,
        use_template=False,
    ):
        """Write one YAML per design plus a manifest; return the manifest path.

        `use_template` is ignored: the calibration is template-free.
        """
        if use_template:
            print(
                "[WARN] BoltzBackend ignores use_template: the calibrated "
                "configuration is template-free."
            )
        if not orig_seqs:
            raise ValueError("orig_seqs is required to build the target chains")

        # orig_seqs is a LIST of entity dicts and each entity carries its own
        # cached MSA directory. target_msa is an OVERRIDE only: pairing MSAs by
        # directory name would hand chain A the MSA named after source chain B.
        target_chains = target_chains_from_orig_seqs(
            orig_seqs,
            target_msa_override=self._get("target_msa", {}) or None,
            a3m_name=self._get("a3m_name", "non_pairing.a3m"),
        )
        self._target_chains = target_chains
        binder_id = self._binder_id([c["id"] for c in target_chains])

        # Validate the pocket contacts here, before any folding: the chain ids
        # are the YAML's, which target_chains_from_orig_seqs renamed from
        # orig_seqs', so the wrong spelling is easy to pass and must not cost a
        # run's GPU time to discover.
        pocket_contacts = self._pocket_contacts()
        if pocket_contacts:
            validate_pocket_contacts(pocket_contacts, target_chains)

        # Validate the cached MSAs here too, and for the same reason: a cached
        # MSA that does not match the sequence is silently replaced by boltz
        # with a dummy, so the run completes, scores badly, and looks like a
        # weak design rather than a missing alignment. Measured: one K->N
        # substitution discarded 3,109 sequences. This is the single chokepoint
        # every fold path passes through, so no caller can forget it.
        #
        # `target_chains`, NOT orig_seqs: this is after the resolver, so the
        # pairs checked are the ones written to the YAML - with any target_msa
        # override applied, the configured a3m_name used, and a string-valued
        # msa already turned into a path.
        msa_records = validate_target_chains(
            target_chains,
            min_depth=int(self._get("min_msa_depth", DEFAULT_MIN_DEPTH)),
        )

        yaml_dir = os.path.join(dump_dir or ".", "yamls")
        entries = []
        for item in data_list:
            name = sample_name_of(item)
            path = write_design_yaml(
                os.path.join(yaml_dir, f"{name}.yaml"),
                target_chains,
                binder_id,
                item["sequence"],
                # [] from the config means unconstrained; the writer reserves
                # an empty list for "you asked to constrain nothing".
                pocket_contacts=pocket_contacts or None,
                pocket_max_distance=float(
                    self._get("pocket_max_distance", DEFAULT_POCKET_MAX_DISTANCE)
                ),
            )
            entries.append(
                {"sample_name": name, "yaml": path, "binder_id": binder_id}
            )
        self._entries = entries
        # Written after the YAMLs so a validation failure leaves no partial
        # output at all. The directory is created here because write_manifest
        # makes its own only AFTER this write, and an empty data_list never
        # reached a YAML that would have created it.
        os.makedirs(dump_dir or ".", exist_ok=True)
        with open(
            os.path.join(dump_dir or ".", "msa_validation.json"), "w"
        ) as fh:
            json.dump(msa_records, fh, indent=2)
        return write_manifest(
            os.path.join(dump_dir or ".", "boltz_manifest.json"), entries
        )

    def predict(
        self,
        input_json_path,
        design_pdb_dir="",
        data_list=None,
        dump_dir="",
        seed=None,
        N_sample=1,
        N_step=2,
        step_scale_eta=1.0,
        gamma0=0.0,
        N_cycle=4,
        binder_chain_idx=None,
        use_msa=True,
        suffix="",
        **kw,
    ):
        """Score every design, gate-confirm the winners, mutate `data_list`.

        N_sample / N_step / step_scale_eta / gamma0 / N_cycle are Protenix
        diffusion parameters with no Boltz equivalent; accepted and ignored.
        Seed depth comes from cfg.seeds_rank / cfg.seeds_gate.
        """
        data_list = data_list or []
        manifest = json.load(open(input_json_path))["designs"]
        self.batch_id = self._batch_id(data_list)

        # --- tier 1: rank pass, ONE batched invocation per seed -----------
        scored = self._score_entries(
            manifest, dump_dir, self.seeds_rank, tag=RANK_TAG, with_sd=False
        )

        # --- pick the winners --------------------------------------------
        keys = [self.ranking_key]
        if self._get("compare_ranking_keys", False):
            keys = ["bz_ipsae", "bz_ipdae"]
        # Eligibility first: an ineligible design must not consume its
        # backbone's confirmation slot, because enforcement at export cannot
        # give the slot back.
        eligible_names = self._policy_eligible_names(scored)
        winners = set()
        for key in keys:
            winners |= self._argmax_per_backbone(
                data_list, scored, key, eligible_names
            )
        if eligible_names is not None:
            print(
                f"    policy: {len(eligible_names)}/{len(scored)} designs "
                f"eligible; {len(winners)} winners confirmed"
            )

        # --- tier 2: gate pass, ONE batched invocation per seed -----------
        gate_entries = [e for e in manifest if e["sample_name"] in winners]
        confirmed = self._score_entries(
            gate_entries, dump_dir, self.seeds_gate, tag=GATE_TAG, with_sd=True
        )

        # --- emit ---------------------------------------------------------
        pred_paths = {}
        for item in data_list:
            name = sample_name_of(item)
            if name in confirmed:
                values = dict(confirmed[name])
            else:
                values = dict(scored.get(name, self._none_row()))
                values.setdefault("bz_n_seeds", self.seeds_rank)
            # bz_batch_id, bz_final_batch_id, bz_pass_tag and bz_struct_path are
            # already set by _score_entries, with a PER-PASS final batch id.
            # Do not collapse them to one id here.
            values.setdefault("bz_batch_id", self.batch_id)
            values.setdefault("bz_final_batch_id", f"{self.batch_id}_unscored")
            values["design_uid"] = design_uid(item)
            item.update(values)
            pred_paths[name] = values.get("bz_struct_path")

        self._write_calibration(dump_dir, data_list)
        return pred_paths

    # -- scoring --------------------------------------------------------- #
    def score_flat(self, entries, dump_dir, seeds, batch_tag="final"):
        """Score every entry at `seeds` depth with no winner selection.

        Task 11's seam: the final common batch needs equal-depth scores for the
        whole shortlist, which the two-tier `predict` cannot provide.

        It establishes its OWN batch id and stamps it on every row. Without
        that, rows keep their per-shard `bz_final_batch_id` and the final
        ranking refuses them - the batch the rescore exists to create would
        never be recorded.
        """
        entries = list(entries)
        self.batch_id = self._batch_id(
            [{"name": e["sample_name"], "seq_idx": 0} for e in entries]
        )
        return self._score_entries(
            entries, dump_dir, seeds, tag=batch_tag, with_sd=True
        )

    def _seed_list(self, seeds, tag):
        """Which seeds this pass folds at.

        Default is `range(1, seeds+1)`, byte-exact with every prior run. But
        that makes the gate pass reuse seed 1, which already screened the
        design - so a winner picked on seed 1 is partly confirming itself, and
        the confirmation rate is inflated by an unknown amount (CLAUDE.md s7.2
        records that the rate is unmeasured).

        `disjoint_confirmation_seeds: true` withholds the screening seeds from
        confirmation, so the gate pass is an independent test. An explicit
        `seeds_rank_list` / `seeds_gate_list` overrides both.
        """
        explicit = self._get(f"seeds_{tag}_list", None)
        if explicit:
            return [int(v) for v in explicit]
        seeds = int(seeds)
        if tag == GATE_TAG and self._get("disjoint_confirmation_seeds", False):
            start = int(self.seeds_rank) + 1
            return list(range(start, start + seeds))
        return list(range(1, seeds + 1))

    def _score_entries(self, entries, dump_dir, seeds, tag, with_sd):
        """Fold `entries` at each seed in ONE invocation per seed, then average.

        Invocation count is bounded by `seeds`, never by len(entries). `tag`
        names the pass and MUST reach _fold_batch: the rank and gate passes both
        use seed 1, so sharing a staging directory would make the gate pass
        refold every rank-pass design still sitting in it.
        """
        entries = list(entries)
        if not entries:
            return {}
        seed_list = self._seed_list(seeds, tag)
        workers = max(1, min(int(self._get("workers", 1)), len(seed_list)))
        if workers == 1:
            per_seed = [
                self._fold_batch(entries, seed, dump_dir, tag=tag)
                for seed in seed_list
            ]
        else:
            # Concurrency is BY SEED, never by design. Each seed folds the FULL
            # entry list, exactly as it would serially, so every invocation has
            # the same batch composition and the scores are unchanged. Sharding
            # designs instead was measured to change scores by ~7x the known
            # cross-batch shift (the same sequence scored ipSAE 0.083 alone and
            # 0.520 in a batch of 4), because a Boltz score depends on what else
            # shares its invocation.
            per_seed = [None] * len(seed_list)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._fold_batch, entries, seed, dump_dir, tag
                    ): idx
                    for idx, seed in enumerate(seed_list)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        per_seed[idx] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[WARN] boltz seed {seed_list[idx]} raised: {exc!r}"
                        )
                        per_seed[idx] = {
                            e["sample_name"]: dict(
                                self._none_row(), bz_status="seed_error"
                            )
                            for e in entries
                        }
        out = {}
        for entry in entries:
            name = entry["sample_name"]
            # Stamp the seed each row came from. aggregate_policy keys its
            # completeness check on row["seed"]; unstamped rows all collapse
            # to the key None, so a 3-seed batch looks like one seed and every
            # verdict degrades to unevaluable.
            rows = []
            for row_seed, batch in zip(seed_list, per_seed):
                row = (batch or {}).get(name)
                rows.append(None if row is None else dict(row, seed=row_seed))
            merged = self._mean_scores(
                rows,
                with_sd=with_sd,
                requested_seeds=seed_list,
                has_enforced_requirements=self._has_enforced_requirements(),
            )
            merged["bz_n_seeds"] = int(seeds)
            # Per-PASS batch id. The rank pass (1 seed) and the gate pass
            # (3 seeds) are different co-fold batches; sharing one id would let
            # the triage pass treat them as comparable and sort 1-seed against
            # 3-seed scores while choosing the shortlist.
            merged["bz_batch_id"] = self.batch_id
            merged["bz_final_batch_id"] = f"{self.batch_id}_{tag}"
            merged["bz_pass_tag"] = tag
            # Taken from the FIRST seed's row, not recomputed: with workers > 1
            # the path depends on which worker folded this design, and only that
            # worker knows.
            first = rows[0] if rows else None
            path = (first or {}).get("bz_struct_path")
            merged["bz_struct_path"] = (
                path if isinstance(path, str) and os.path.isfile(path) else None
            )
            out[name] = merged
        return out

    def _batch_digest(self, entries, seed, tag):
        """Everything that changes a score, in one hash.

        Composition is included because a Boltz score depends on which other
        designs shared its invocation (CLAUDE.md s1): the same sequence scored
        0.082758 alone and 0.519555 in a batch of four. So a batch whose
        membership changed is a different measurement, not a resumable one.

        The YAML's TEXT is not enough. It names an MSA by path, so editing a
        cached a3m in place leaves every YAML byte-identical while changing
        what the predictor conditions on. The referenced MSA contents and the
        predictor's own identity are therefore hashed too.

        The predictor's identity is the executable FILE (path, size, mtime),
        not the library version. The `boltz` entry point is a tiny shim over
        `boltz.main`, so a `pip install -U boltz` can leave it unchanged.
        """
        h = hashlib.sha256()
        h.update(f"{tag}\x00{seed}\x00".encode())
        for entry in sorted(entries, key=lambda e: e["sample_name"]):
            h.update(entry["sample_name"].encode())
            h.update(b"\x00")
            h.update(file_sha256(entry["yaml"]).encode())
            h.update(b"\x00")
        # The MSAs the YAMLs point AT, by content.
        for chain in self._target_chains or []:
            h.update(f"{chain['id']}\x00".encode())
            try:
                h.update(file_sha256(chain["msa"]).encode())
            except (OSError, KeyError, TypeError):
                h.update(b"unreadable")
            h.update(b"\x00")
        # The predictor itself: a different boltz is a different measurement.
        try:
            boltz_bin = br.resolve_boltz_bin(self._get("boltz_bin", None))
            st = os.stat(boltz_bin)
            h.update(f"{boltz_bin}:{st.st_size}:{int(st.st_mtime)}\x00".encode())
        except Exception:  # noqa: BLE001
            h.update(b"boltz_unresolved\x00")
        for key in (
            "recycling_steps", "diffusion_samples", "boltz_num_workers",
            "boltz_no_kernels", "pocket_max_distance",
        ):
            h.update(f"{key}={self._get(key, None)}\x00".encode())
        return h.hexdigest()

    def _fold_batch(self, entries, seed, dump_dir, tag="fold"):
        """ONE `boltz predict` over the WHOLE entry list, at one seed.

        The only site that launches Boltz. The batching unit is a directory:
        check_inputs (boltz/main.py:281) expands it with glob("*") and
        process_inputs (main.py:795) folds every path it returns, so the model
        loads once per invocation rather than once per design.

        NEVER sharded across processes. A design's Boltz score depends on which
        other designs share its invocation: measured, the same sequence scored
        bz_ipsae 0.082758 folded alone and 0.519555 folded in a batch of four,
        while reordering the other three changed nothing to six decimals. Batch
        composition, not ordering, is what moves the number - so splitting a
        batch would silently change every score in it. Concurrency lives one
        level up in _score_entries, which runs the per-SEED folds in parallel;
        each of those still folds the full entry list.

        `tag` separates passes. Without it the rank pass (seed 1) and the gate
        pass (seeds 1..3) would share boltz_pred/seed_1/input, and since Boltz
        folds EVERY yaml in the directory the gate pass would refold all the
        rank-only designs - inflating the budget and changing the composition.

        The out_dir is the seed root, never the input dir: check_inputs
        (main.py:300-311) raises on a subdirectory inside the input directory.

        Overridden in tests; keep it the single seam.
        """
        seed_root = self._seed_root(dump_dir, tag, seed)
        yaml_dir = os.path.join(seed_root, bp.BATCH_INPUT_STEM)
        os.makedirs(yaml_dir, exist_ok=True)
        wanted = {e["sample_name"] for e in entries}
        # a stale yaml would silently widen the batch
        for stale in os.listdir(yaml_dir):
            if stale.endswith(".yaml") and stale[: -len(".yaml")] not in wanted:
                os.remove(os.path.join(yaml_dir, stale))
        for entry in entries:
            staged = os.path.join(yaml_dir, f"{entry['sample_name']}.yaml")
            # ALWAYS refresh. Copying only when absent meant that reusing an
            # output directory after changing a sequence or a target folded the
            # previous input: the run succeeded, the scores were real, and they
            # belonged to a design nobody asked about.
            with open(entry["yaml"]) as src, open(staged, "w") as dst:
                dst.write(src.read())

        # A rerun whose inputs changed must not read the previous run's
        # predictions. The layout stays as it is - a test pins it - so the
        # digest is recorded here and a mismatch clears the stale outputs
        # rather than quietly reusing them.
        digest = self._batch_digest(entries, seed, tag)
        record_path = os.path.join(seed_root, ".batch_config.json")
        previous = None
        if os.path.exists(record_path):
            try:
                with open(record_path) as fh:
                    rec = json.load(fh)
                # Valid JSON that is not an object ([], null, 3) is as
                # unverifiable as broken JSON; .get on it would crash the fold.
                previous = rec.get("digest") if isinstance(rec, dict) else None
            except (ValueError, OSError):
                # A missing or unreadable record means the outputs beside it
                # are UNVERIFIED, which is the same as stale. Absence of
                # evidence is not evidence that they match.
                previous = None
        # Clearing boltz_results_* is what INVALIDATES BOLTZ'S INPUT CACHE, and
        # it must stay unconditional. boltz/main.py:724-742 converts only the
        # records NOT already in <out_dir>/processed/records ("Found N existing
        # processed inputs, skipping them"), independent of --override, and
        # main.py:1134 puts out_dir under boltz_results_<stem>, so processed/
        # lives inside the tree removed below. Refreshing the staged YAML above
        # is therefore INERT on its own: boltz would re-predict the previous
        # processed record and report `ok`. Making this clearing conditional
        # (say, "keep the tree when the digest matches") re-opens the
        # stale-staged-YAML defect that the always-refresh closed. The digest
        # only decides what to SAY, not whether to clear.
        if previous != digest and previous is not None:
            print(
                f"[WARN] {seed_root}: the evaluation configuration changed "
                f"({previous[:12]} -> {digest[:12]}). The predictions here are "
                f"from different inputs."
            )
        elif previous is None and os.path.isdir(seed_root):
            if any(c.startswith("boltz_results_")
                   for c in os.listdir(seed_root)):
                print(
                    f"[WARN] {seed_root}: predictions present with no "
                    f"verifiable configuration record; treating them as stale."
                )
        for child in os.listdir(seed_root) if os.path.isdir(seed_root) else []:
            if child.startswith("boltz_results_"):
                shutil.rmtree(os.path.join(seed_root, child),
                              ignore_errors=True)
        with open(record_path, "w") as fh:
            json.dump(
                {
                    "digest": digest,
                    "entries": sorted(e["sample_name"] for e in entries),
                    "params": {
                        "tag": tag,
                        "seed": seed,
                        "recycling_steps": self._get("recycling_steps", 3),
                        "diffusion_samples": self._get("diffusion_samples", 1),
                    },
                },
                fh,
                indent=2,
            )

        result = br.run_seed(
            br.resolve_boltz_bin(self._get("boltz_bin", None)),
            yaml_dir,
            seed_root,
            seed,
            timeout_s=int(self._get("timeout_s", 3600)),
            gpu=str(self._get("gpu", "0")),
            recycling_steps=int(self._get("recycling_steps", 3)),
            diffusion_samples=int(self._get("diffusion_samples", 1)),
            num_workers=int(self._get("boltz_num_workers", 0)),
            no_kernels=bool(self._get("boltz_no_kernels", True)),
        )
        # Boltz replaces a non-matching MSA with a dummy, says so once on
        # stdout, and exits 0. prepare_json should have caught this already;
        # if it reaches here the inputs changed underneath us, so it is an
        # error and not a warning. This is checked BEFORE returncode on purpose:
        # a discarded MSA is the more specific diagnosis, and a run that
        # discarded one and then also failed should report the cause, not just
        # `boltz_failed`.
        if result.get("msa_dummy_count"):
            # Counted over the FULL output stream by run_seed, not matched
            # against a tail: the warning appears early and later progress
            # output displaces it from any fixed-size window.
            print(
                f"[ERROR] boltz seed {seed} DISCARDED a cached MSA "
                f"({result['msa_dummy_count']} occurrence(s)) and predicted "
                f"unconditioned (counted per design x chain, not per chain). "
                f"Every score from this seed is meaningless. prepare_json "
                f"validated these same MSAs at the start of this run, so "
                f"something changed underneath it: compare the a3m files and "
                f"sequences against the hashes recorded at validation time in "
                f"{os.path.join(dump_dir or '.', 'msa_validation.json')}. "
                f"Boltz said:\n  "
                + "\n  ".join(result.get("msa_dummy_lines") or [])
            )
            return {
                entry["sample_name"]: dict(
                    self._none_row(), bz_status="msa_discarded"
                )
                for entry in entries
            }

        if result["returncode"] != 0:
            status = (
                "boltz_timeout" if result["returncode"] == 124 else "boltz_failed"
            )
            print(
                f"[WARN] boltz seed {seed} failed: "
                f"\n{self._boltz_output(result, 800)}"
            )
            return {
                entry["sample_name"]: dict(self._none_row(), bz_status=status)
                for entry in entries
            }

        # One invocation wrote every prediction; parse them all from it.
        scored = {}
        for entry in entries:
            name = entry["sample_name"]
            pred_dir = bp.prediction_dir(
                seed_root, name, input_stem=bp.BATCH_INPUT_STEM
            )
            row = bp.score_prediction(
                pred_dir,
                name,
                entry.get("binder_id", "C"),
                ipdae_cutoff=float(self._get("ipdae_cutoff", 8.0)),
                ipsae_pae_cutoff=float(self._get("ipsae_pae_cutoff", 10.0)),
                epitope_policy=self._get("epitope_policy", {}) or None,
            )
            # the worker that folded it is the only one that knows where it is
            pdb = os.path.join(pred_dir, f"{name}_model_0.pdb")
            row["bz_struct_path"] = pdb if os.path.isfile(pdb) else None
            scored[name] = row

        # `State=COMPLETED` is not success, one level down: boltz exits 0 even
        # when every example fails, reporting it only as "Number of failed
        # examples: N" on stdout. Checking the return code alone turned 64
        # failed folds into 64 silent `missing_pdb` rows with no diagnostic
        # anywhere - the cause (another job holding the GPU) was only
        # recoverable by comparing file timestamps afterwards. A seed that
        # produced NOTHING is reported here, with boltz's own output, because
        # the alternative is a campaign that looks merely unlucky.
        # Counts come from each row's own bz_status, never from a second,
        # hand-written list of required files. parse.py (read_prediction) needs
        # only the PDB and pae_*.npz - plddt and confidence_*.json are optional -
        # so a separate file check was stricter than the parser, and called a
        # prediction "not produced" on the very line beside an `ok` row.
        with_pdb = sum(1 for r in scored.values() if r.get("bz_struct_path"))
        produced = sum(1 for r in scored.values() if r.get("bz_status") == "ok")
        incomplete = [
            (name, row.get("bz_status"))
            for name, row in scored.items()
            if row.get("bz_struct_path") and row.get("bz_status") != "ok"
        ]
        # Three distinct states, because each has a different cause and the
        # diagnosis for one is wrong for the others: no PDB at all (GPU
        # contention, MSA mismatch), PDBs with no usable score (a run cut
        # short or an output-format change - structures DO exist), or a mix.
        if entries and not with_pdb:
            print(
                f"[WARN] boltz seed {seed} returned 0 but wrote NO structures "
                f"for any of {len(entries)} entries. Every row will be "
                f"`missing_pdb`. Boltz reports per-example failures on stdout "
                f"and still exits 0, so check for GPU contention (it needs the "
                f"device to itself) and for an MSA that does not match the "
                f"sequence. Boltz's own output:\n"
                f"{self._boltz_output(result)}"
            )
        elif entries and not produced:
            print(
                f"[WARN] boltz seed {seed} wrote {with_pdb} structure(s) but "
                f"none could be scored, e.g. {incomplete[0][0]} has status "
                f"`{incomplete[0][1]}`. These are failures, not weak "
                f"designs. Boltz's own output:\n"
                f"{self._boltz_output(result)}"
            )
        elif entries and produced < len(entries):
            clauses = []
            if len(entries) - with_pdb:
                clauses.append(
                    f"{len(entries) - with_pdb} wrote no structure "
                    f"(`missing_pdb`)"
                )
            if incomplete:
                clauses.append(
                    f"{len(incomplete)} wrote a PDB that could not be scored, "
                    f"e.g. {incomplete[0][0]} has status `{incomplete[0][1]}` "
                    f"(failures, not weak designs)"
                )
            print(
                f"[WARN] boltz seed {seed}: only {produced}/{len(entries)} "
                f"entries produced a scoreable structure; "
                + "; ".join(clauses) + "."
            )
        return scored

    # -- internals ------------------------------------------------------- #
    def _binder_id(self, target_ids):
        configured = self.cfg.get("binder_chain_id")
        if configured:
            return configured
        for candidate in "CDEFGHIJKLMNOP":
            if candidate not in target_ids:
                return candidate
        raise ValueError(f"no free binder chain id given targets {target_ids}")

    @staticmethod
    def _boltz_output(result, limit=1200):
        """Boltz's own diagnostics, with nothing displaced by anything else.

        The 'Number of failed examples' line is extracted from the FULL stream
        by run_seed and printed first; the two streams then follow as separate
        labelled blocks. A single concatenated tail put stdout first, so any
        stderr longer than the window pushed the stdout diagnostic out.
        """
        parts = []
        failed = result.get("failed_example_lines") or []
        if failed:
            parts.append("  " + "\n  ".join(failed))
        out_tail = (result.get("stdout_tail") or "")[-limit:]
        err_tail = (result.get("stderr_raw_tail") or "")[-limit:]
        parts.append(f"--- stdout (tail) ---\n{out_tail}")
        parts.append(f"--- stderr (tail) ---\n{err_tail}")
        return "\n".join(parts)

    def _none_row(self):
        row = {key: None for key in MEAN_KEYS}
        row["bz_status"] = "not_scored"
        row["bz_struct_path"] = None
        return row

    def _seed_root(self, dump_dir, tag, seed):
        """Where one seed's boltz invocation writes.

        One directory per (tag, seed). There is deliberately no per-worker
        subdivision: parallelism is by seed, and seeds already have separate
        directories. A per-design sharding layout existed briefly and was
        removed, because splitting a batch changes every score in it.
        """
        return os.path.join(
            dump_dir or ".", "boltz_pred", str(tag), f"seed_{seed}"
        )

    def _structure_path(self, dump_dir, sample_name, seed, tag="fold"):
        """The PDB file, not its directory. Task 9 copies this into the export.

        Layout follows _fold_batch: boltz_pred/<tag>/seed_<n>/. The path is
        persisted per row as bz_struct_path, because the export stage cannot
        reconstruct which dump_dir and tag produced a given row.
        """
        return os.path.join(
            bp.prediction_dir(
                os.path.join(dump_dir or ".", "boltz_pred", str(tag), f"seed_{seed}"),
                sample_name,
                input_stem=bp.BATCH_INPUT_STEM,
            ),
            f"{sample_name}_model_0.pdb",
        )

    @staticmethod
    def _mean_scores(rows, with_sd=False, requested_seeds=None,
                     has_enforced_requirements=False):
        """Mean over seeds. Any failed seed makes the design unusable.

        Policy fields are carried through explicitly (R1). This method used to
        rebuild the row from MEAN_KEYS alone, which dropped every ep_* column
        before selection, so a gate clause on ep_satisfied read a column that
        did not exist. The policy verdict is a tri-state conjunction, never a
        mean of booleans.
        """
        rows = [r for r in rows if r]
        seeds = tuple(
            requested_seeds
            if requested_seeds is not None
            else [r.get("seed") for r in rows]
        )
        policy = aggregate_policy(
            rows, seeds, has_enforced_requirements=has_enforced_requirements
        )
        bad = [r for r in rows if r.get("bz_status") != "ok"]
        if not rows or bad:
            out = {key: None for key in MEAN_KEYS}
            out["bz_status"] = bad[0]["bz_status"] if bad else "not_scored"
            if with_sd:
                out["bz_ipsae_sd"] = None
                out["bz_ipdae_sd"] = None
            out.update(policy)
            return out
        out = {"bz_status": "ok"}
        out.update(policy)
        for key in MEAN_KEYS:
            values = [r[key] for r in rows if r.get(key) is not None]
            out[key] = float(np.mean(values)) if values else None
        if with_sd:
            for key in ("bz_ipsae", "bz_ipdae"):
                values = [r[key] for r in rows if r.get(key) is not None]
                out[f"{key}_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        return out

    def _has_enforced_requirements(self):
        """Did the CONFIG declare an epitope policy to enforce?

        Manifest fact, never inferred from a result: the metric returns
        `ep_satisfied=None` for both "no policy" and "unevaluable", so a value
        cannot distinguish them (metrics/epitope.py:180).
        """
        policy = self._get("epitope_policy", {}) or {}
        return bool(policy.get("required") or policy.get("forbidden"))

    def _policy_eligible_names(self, scored):
        """Sample names the policy permits, or None when no policy is declared.

        None means "do not filter", which keeps the legacy path byte-exact.
        """
        if not self._has_enforced_requirements():
            return None
        from pxdbench.targets.eligibility import status_permits

        return {
            name
            for name, row in scored.items()
            if status_permits((row or {}).get("policy_status"))
        }

    @staticmethod
    def _argmax_per_backbone(data_list, scored, key, eligible_names=None,
                             tie_tolerance=IPSAE_TIE_TOLERANCE):
        """Best scoring design per backbone, among those the policy permits.

        `eligible_names=None` disables the filter and is the legacy behaviour.

        Without the filter this ranked on score alone, so an off-site sibling
        that scored well could take its backbone's single confirmation slot
        from an on-site sibling that scored slightly lower - and epitope
        engagement anti-correlates with ipSAE here (rho -0.227, CLAUDE.md
        s7.1), so that is the common case, not the rare one. The budget is
        spent before enforcement at export ever gets a say.
        """
        per_backbone = {}
        for item in data_list:
            name = sample_name_of(item)
            if eligible_names is not None and name not in eligible_names:
                continue
            value = (scored.get(name) or {}).get(key)
            if value is None:
                continue
            per_backbone.setdefault(item["name"], []).append((name, value))

        winners = set()
        for candidates in per_backbone.values():
            top = max(v for _n, v in candidates)
            # Everything within the measured noise band is TIED. Picking the
            # nominal maximum here discards siblings on a difference the
            # instrument cannot resolve (signal-to-noise 1.88), and epitope
            # engagement anti-correlates with ipSAE, so the nominal winner is
            # if anything the likelier off-site one.
            tied = [
                (n, v) for n, v in candidates
                if top - v <= tie_tolerance
            ]
            if len(tied) > 1:
                tied.sort(key=lambda nv: (
                    -_engagement_of(scored.get(nv[0])),
                    -nv[1],
                    nv[0],
                ))
            else:
                tied.sort(key=lambda nv: -nv[1])
            winners.add(tied[0][0])
        return winners

    def _batch_id(self, data_list):
        digest = hashlib.sha256(
            "|".join(sorted(sample_name_of(i) for i in data_list)).encode()
        ).hexdigest()[:12]
        return f"{int(time.time())}_{digest}"

    def _write_calibration(self, dump_dir, data_list):
        boltz_bin = br.resolve_boltz_bin(self._get("boltz_bin", None))
        """Provenance without which a result is not interpretable."""
        pocket_contacts = self._pocket_contacts()
        record = {
            "gate": GATE,
            "gate_name": GATE_NAME,
            "gate_is_provisional": True,
            # A pocket-constrained run's bz_* scores are not the scores GATE
            # was calibrated on. Without this flag such a run's numbers look
            # exactly like an unconstrained run's.
            "gate_applies": not pocket_contacts,
            "pocket_contacts": [list(c) for c in pocket_contacts],
            "pocket_max_distance": (
                float(self._get("pocket_max_distance", DEFAULT_POCKET_MAX_DISTANCE))
                if pocket_contacts
                else None
            ),
            "ranking_key": self.ranking_key,
            "compare_ranking_keys": bool(self._get("compare_ranking_keys", False)),
            "seeds_rank": self.seeds_rank,
            "seeds_gate": self.seeds_gate,
            "aggregation": "mean",
            "recycling_steps": int(self._get("recycling_steps", 3)),
            "diffusion_samples": int(self._get("diffusion_samples", 1)),
            "ipdae_cutoff": float(self._get("ipdae_cutoff", 8.0)),
            "ipsae_pae_cutoff": float(self._get("ipsae_pae_cutoff", 10.0)),
            # EXPECTED is a constant in parse.py; MEASURED asks the interpreter
            # that sits beside the boltz binary. Recording only the constant
            # made this file claim a version nothing had checked.
            "boltz_version_expected": bp.BOLTZ_VERSION_EXPECTED,
            "boltz_version_measured": measured_boltz_version(boltz_bin),
            "boltz_bin": boltz_bin,
            # keyed by DESIGN chain id, resolved from the entities themselves
            "target_msa": {
                chain["id"]: {
                    "path": chain["msa"],
                    # boltz dedups (boltz/data/parse/a3m.py:57-62), so the
                    # header count overstates what it loads; both are kept,
                    # named as in msa_validation.json.
                    **dict(zip(("depth_headers", "depth_unique"),
                               mc.a3m_depths(chain["msa"]))),
                }
                for chain in self._target_chains
            },
            "target_msa_override": self._get("target_msa", {}) or {},
            "anchor_sequence": self._get("anchor_sequence", ""),
            "epitope_policy": self._get("epitope_policy", {}) or {},
            "batch_id": self.batch_id,
            "n_designs": len(data_list),
            "common_scorer_sha256": self._common_scorer_hash(),
        }
        os.makedirs(dump_dir or ".", exist_ok=True)
        with open(os.path.join(dump_dir or ".", "bz_calibration.json"), "w") as handle:
            json.dump(_jsonable(record), handle, indent=2)

    @staticmethod
    def _common_scorer_hash():
        """The hash of the scorer ACTUALLY LOADED, not of a configured path.

        This hashed `interface.COMMON_SCORER_DIR/common_scorer.py` - a path
        that need not be the implementation in use, now that the bundled copy
        is the default and an explicit override may point elsewhere. A scorer
        hash change invalidates evaluation-context identity even when golden
        values agree.
        """
        try:
            return bp.im.scorer_provenance()["scorer_sha256"] or None
        except (AttributeError, OSError):
            return None
