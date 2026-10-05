"""Residue numbering, chain mapping and hotspot resolution - pure stdlib.

Extracted verbatim from `pxdesign/utils/infer.py` so that the map and role
contracts can be tested without initialising a model. Importing `infer`
pulls in torch, protenix, biotite and ml_collections (measured: 1.3 s and four
heavy roots), which the targeting-contract tests must not require.

The algorithms are NOT duplicated: `infer` re-exports every public name from
here, so there is exactly one implementation and existing import paths keep
working. `tests/test_residue_numbering_extraction.py` pins that equivalence.

Depends on nothing beyond the standard library.
"""
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# -------------------------
# Handling PDB input
# -------------------------


def parse_ranges(range_str: str) -> list[tuple[int, int]]:
    """
    Parse "1-30,40-50,66" -> [(1,30),(40,50),(66,66)]
    """
    ranges: list[tuple[int, int]] = []
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ranges.append((int(a), int(b)))
        else:
            x = int(part)
            ranges.append((x, x))
    return ranges


def format_ranges(ints: Iterable[int]) -> str:
    """
    Compress sorted integers into "a-b,c,d-e".
    """
    xs = sorted(set(int(x) for x in ints))
    if not xs:
        return ""
    out: list[str] = []
    s = e = xs[0]
    for x in xs[1:]:
        if x == e + 1:
            e = x
        else:
            out.append(f"{s}-{e}" if s != e else f"{s}")
            s = e = x
    out.append(f"{s}-{e}" if s != e else f"{s}")
    return ",".join(out)


# -------------------------
# Chain mapping
# -------------------------


def build_chain_mapping(
    old_ids: Iterable[str],
    new_ids: Iterable[str],
    *,
    keep_chains: Iterable[str] | None = None,
    err_hint: str = "Please consider using a CIF structure file in your JSON file.",
) -> dict[str, str]:
    """
    Build mapping old_id -> new_id with consistency check.

    keep_chains:
      - None: keep all chains
      - otherwise: only build mapping for chains in keep_chains
    """
    old_ids = list(old_ids)
    new_ids = list(new_ids)
    if len(old_ids) != len(new_ids):
        raise ValueError("old_ids and new_ids must have the same length.")

    keep = set(keep_chains) if keep_chains is not None else None

    mapping: dict[str, str] = {}
    for old, new in zip(old_ids, new_ids):
        if keep is not None and old not in keep:
            continue
        if old not in mapping:
            mapping[old] = new
        elif mapping[old] != new:
            raise ValueError(
                f"Inconsistent mapping: chain '{old}' maps to both "
                f"'{mapping[old]}' and '{new}'. It will raise ambiguity. {err_hint}"
            )
    return mapping


# -------------------------
# Residue mapping (res_id <-> auth_res_id)
# -------------------------


@dataclass(frozen=True)
class ResidueMaps:
    # (chain_id, res_id) -> (auth_asym_id, auth_res_id)
    resid2auth: dict[tuple[str, int], tuple[str, int]]
    # (auth_asym_id, auth_res_id) -> (chain_id, res_id)
    auth2resid: dict[tuple[str, int], tuple[str, int]]


def build_residue_maps(
    atom_array,
    *,
    strict_bijective: bool = True,
    err_hint: str = "Please consider using a CIF structure file in your JSON file.",
) -> ResidueMaps:
    """
    Build residue-level mapping with uniqueness checks.

    Ensures:
      - each (chain_id, res_id) maps to a single (auth_asym_id, auth_res_id)
      - (optional) bijection: each (auth_asym_id, auth_res_id) maps back to a single (chain_id, res_id)
    """
    chain_id = atom_array.chain_id
    res_id = atom_array.res_id
    auth_asym_id = atom_array.auth_asym_id
    auth_res_id = atom_array.auth_res_id

    resid2auth: dict[tuple[str, int], tuple[str, int]] = {}
    auth2resid: dict[tuple[str, int], tuple[str, int]] = {}

    for c, r, ac, ar in zip(chain_id, res_id, auth_asym_id, auth_res_id):
        key = (str(c), int(r))
        val = (str(ac), int(ar))

        if key in resid2auth and resid2auth[key] != val:
            raise ValueError(
                "Non-unique mapping detected: same (chain_id, res_id) maps to multiple "
                f"(auth_asym_id, auth_res_id).\n  key={key}\n  first={resid2auth[key]}\n  new={val}\n"
                f"{err_hint}"
            )
        resid2auth.setdefault(key, val)

        if strict_bijective:
            if val in auth2resid and auth2resid[val] != key:
                raise ValueError(
                    "Non-unique mapping detected: same (auth_asym_id, auth_res_id) maps to multiple "
                    f"(chain_id, res_id).\n  val={val}\n  first={auth2resid[val]}\n  new={key}\n"
                    f"{err_hint}"
                )
            auth2resid.setdefault(val, key)

    return ResidueMaps(resid2auth=resid2auth, auth2resid=auth2resid)


# -------------------------
# Converters: crop / hotspot
# -------------------------


def convert_crop_auth_to_new(
    crop_dict: dict[str, str],
    residue_maps: ResidueMaps,
    *,
    strict_mapping: bool = True,
) -> dict[str, str]:
    result: dict[str, list[int]] = defaultdict(list)

    for auth_chain, range_str in crop_dict.items():
        for start, end in parse_ranges(range_str):
            for auth_r in range(start, end + 1):
                key = (auth_chain, int(auth_r))
                if key not in residue_maps.auth2resid:
                    if strict_mapping:
                        raise KeyError(
                            f"Requested auth residue not found in atom_array: {key}"
                        )
                    else:
                        continue
                new_c, new_r = residue_maps.auth2resid[key]
                result[new_c].append(new_r)

    return {new_c: format_ranges(rs) for new_c, rs in result.items() if rs}


def convert_hotspot_auth_to_new(
    hotspot_dict: dict[str, list[int]],
    residue_maps: ResidueMaps,
    *,
    strict_mapping: bool = True,
) -> dict[str, list[int]]:
    """
    {chain_id: [11,22]} -> {auth_asym_id: [auth_res_id,...]} (sorted unique)
    """
    result: dict[str, list[int]] = defaultdict(list)

    for chain, res_list in hotspot_dict.items():
        for r in res_list:
            key = (chain, int(r))
            if key not in residue_maps.auth2resid:
                if strict_mapping:
                    raise KeyError(f"Hotspot residue not found in atom_array: {key}")
                else:
                    continue
            ac, ar = residue_maps.auth2resid[key]
            result[ac].append(ar)

    return {ac: sorted(set(ars)) for ac, ars in result.items() if ars}


# -------------------------
# Apply filter rewrite (chain_id / crop / msa / hotspot)
# -------------------------


def rewrite_input_dict_inplace(
    input_dict: dict,
    *,
    chain_mapping: dict[str, str],  # old chain_id -> new chain_id (e.g. PDB->CIF)
    residue_maps: ResidueMaps | None,
) -> None:
    """
    Rewrite cond_dict['filter'] in-place using chain_mapping and residue_maps.
    """
    cond_dict = input_dict["condition"]
    filt = cond_dict.get("filter", {})
    if filt:
        # chain_id list
        if "chain_id" in filt and filt["chain_id"]:
            filt["chain_id"] = [chain_mapping[c] for c in filt["chain_id"]]

        # crop dict: {chain_id: "ranges"}
        if "crop" in filt and filt["crop"]:
            if residue_maps is None:
                raise ValueError(
                    "filter.crop requires residue_maps (atom_array) to convert res_id -> auth_res_id."
                )
            filt["crop"] = convert_crop_auth_to_new(filt["crop"], residue_maps)
        cond_dict["filter"] = filt

    # msa dict: {chain_id: ...}
    if "msa" in cond_dict and cond_dict["msa"]:
        cond_dict["msa"] = {chain_mapping[k]: v for k, v in cond_dict["msa"].items()}

    input_dict["condition"] = cond_dict
    # hotspot dict: {chain_id: [res_ids]}
    if "hotspot" in input_dict and input_dict["hotspot"]:
        if residue_maps is None:
            raise ValueError(
                "filter.hotspot requires residue_maps (atom_array) to convert res_id -> auth_res_id."
            )
        input_dict["hotspot"] = convert_hotspot_auth_to_new(
            input_dict["hotspot"], residue_maps
        )


# -------------------------
# Hotspot resolution and validation
# -------------------------


def _structure_residue_index(atom_array) -> dict:
    """{(chain_id, res_id)} actually present in the structure the model sees."""
    return {
        (str(c), int(r))
        for c, r in zip(atom_array.chain_id, atom_array.res_id)
    }


def validate_hotspots(hotspot_dict: dict, atom_array, *, strict: bool = True) -> dict:
    """Check every requested hotspot exists in the structure's own numbering.

    Hotspots are a SOFT generation condition: json_parser turns them into a
    boolean mask with np.isin(chain_array.res_id, hotspot). A residue number
    that does not exist simply matches nothing, so a typo, a wrong chain or a
    numbering mismatch was silently ignored and the campaign ran unconditioned
    on the residues the operator asked for.

    Returns the dict unchanged when every residue resolves.
    """
    if not hotspot_dict:
        return hotspot_dict
    present = _structure_residue_index(atom_array)
    missing = [
        f"{chain}{res}"
        for chain, residues in hotspot_dict.items()
        for res in (residues or [])
        if (str(chain), int(res)) not in present
    ]
    if missing:
        chains = sorted({str(c) for c, _ in present})
        msg = (
            f"hotspot residues not found in the structure: {missing}. "
            f"Chains present: {chains}. Hotspots must be given in the numbering "
            f"of the structure the model sees; np.isin would otherwise match "
            f"nothing and the design would run unconditioned on them."
        )
        if strict:
            raise KeyError(msg)
        logger.warning(msg)
    return hotspot_dict


def record_resolved_hotspots(input_dict: dict, out_dir: str | None) -> None:
    """Persist the hotspots in the numbering the model and outputs use.

    Without this the epitope check downstream has to guess whether a requested
    residue number refers to the source structure or the design, and a missed
    epitope is indistinguishable from a numbering mismatch.
    """
    if not out_dir or not input_dict.get("hotspot"):
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "resolved_hotspots.json")
        payload = {
            "name": input_dict.get("name"),
            "hotspot_resolved": input_dict["hotspot"],
            "numbering": "design (chain_id, res_id) - the same numbering as the "
                         "exported complexes",
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:  # pragma: no cover - best effort provenance
        logger.warning(f"could not record resolved hotspots: {exc}")


