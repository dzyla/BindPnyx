"""Residue requests that state what they mean, and fail when they are wrong.

`validate_hotspots` only catches residues that are **absent**. A residue that
exists but is the wrong one passes silently - which is precisely why "did we
target His535?" could not be answered: the number resolved, so nothing
complained, and the only way to check was to recover the source structure by
hand.

A spec of `"H535"` carries an assertion: *and the residue there is a
histidine*. A mismatch is a preflight failure, before any GPU work.

Position parsing is exact-integer by requirement (targeting-contracts section
2), not `int(float(x))`. That expression silently turns `"22.9"` into position
22, and a pandas `protect` column holding both blanks and integers types as
float, so `"22.0"` arrives where `22` was meant. Both are rejected rather than
guessed.

Standard library only.
"""
from dataclasses import dataclass

from pxdbench.targets.residue_map import THREE_TO_ONE

ONE_TO_THREE = {one: three for three, one in THREE_TO_ONE.items()}


class IdentityAssertionError(ValueError):
    """A spec asserted an identity the structure does not have."""


@dataclass(frozen=True)
class ResidueSpec:
    chain: str | None
    res_id: int
    #: Three-letter code the caller asserted, or None for no assertion.
    expect_resname: str | None


def _exact_int(text, original):
    """A position, or a loud failure. No truncation, no float coercion."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"cannot parse residue spec {original!r}: empty position")
    if not cleaned.isdigit():
        raise ValueError(
            f"cannot parse residue spec {original!r}: {cleaned!r} is not an "
            f"exact integer. Floats are rejected rather than truncated - "
            f"int(float('22.9')) would silently give position 22, and a mixed "
            f"blank/integer column types as float so '22.0' arrives where 22 "
            f"was meant."
        )
    value = int(cleaned)
    if value <= 0:
        raise ValueError(
            f"residue spec {original!r}: position must be positive, got {value}"
        )
    return value


def parse_residue_spec(value, default_chain=None):
    """Parse `535`, `"535"`, `"H535"`, `"D:535"` or `"D:H535"`."""
    if isinstance(value, bool):
        raise ValueError(f"residue spec {value!r} is a boolean, not a position")
    if isinstance(value, float):
        raise ValueError(
            f"residue spec {value!r} is a float; an exact integer is required. "
            f"A mixed blank/integer column types as float, so this is more "
            f"likely a type accident than an intent."
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                f"residue spec {value!r}: position must be positive"
            )
        return ResidueSpec(default_chain, value, None)

    text = str(value).strip()
    chain = default_chain
    if ":" in text:
        chain_part, text = text.split(":", 1)
        chain = chain_part.strip() or default_chain
        if not chain:
            raise ValueError(f"residue spec {value!r}: empty chain")
    text = text.strip()
    if not text:
        raise ValueError(f"cannot parse residue spec {value!r}")

    if text[0].isdigit() or text[0] == "-":
        return ResidueSpec(chain, _exact_int(text, value), None)

    letter, rest = text[0].upper(), text[1:]
    if letter not in ONE_TO_THREE:
        raise ValueError(
            f"residue spec {value!r}: {letter!r} is not a one-letter amino "
            f"acid code. A typo must not silently become an unasserted "
            f"residue - that is the failure this assertion exists to prevent."
        )
    return ResidueSpec(chain, _exact_int(rest, value), ONE_TO_THREE[letter])


def resolve_specs(requests, residue_map, numbering):
    """Resolve {chain: [spec, ...]} into unique sorted ResidueRows.

    `numbering` is declared, never inferred: the two branches of
    `convert_to_bioassembly_dict` give the same number different meanings
    depending on the input file's extension.
    """
    if numbering not in ("auth", "work"):
        raise ValueError(
            f"numbering must be 'auth' or 'work', got {numbering!r}. It is "
            f"declared rather than inferred from a file extension."
        )

    rows = {}
    for chain, specs in (requests or {}).items():
        for raw in specs:
            spec = parse_residue_spec(raw, default_chain=chain)
            lookup = (
                residue_map.to_work if numbering == "auth" else residue_map.to_auth
            )
            row = lookup(spec.chain, spec.res_id)
            if spec.expect_resname and row.resname != spec.expect_resname:
                raise IdentityAssertionError(
                    f"request {raw!r} on chain {spec.chain} asserts "
                    f"{spec.expect_resname}, but "
                    f"{row.auth_chain}{row.auth_res_id} "
                    f"(work {row.work_chain}{row.work_res_id}) is "
                    f"{row.resname}. Refusing to proceed: a wrong-but-present "
                    f"residue is exactly what resolves silently today."
                )
            rows[(row.work_chain, row.work_res_id)] = row
    return [rows[key] for key in sorted(rows)]
