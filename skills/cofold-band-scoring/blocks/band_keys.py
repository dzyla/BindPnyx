#!/usr/bin/env python
"""One place that knows what a band file's terms are called.

Why this exists. The mechanism band is a conjunction of per-target geometric terms - in the campaign
these blocks came from, an anchor salt-bridge distance to one receptor residue, a conserved-core
contact count, and an insertion depth - and the band file's keys were named after that receptor's
residue. Three blocks then hardcoded those names, which quietly made "campaign-agnostic" blocks
specific to one target: on a different receptor they would raise KeyError on a band file that was
perfectly valid.

So each logical term resolves through a candidate list. The GENERIC name comes first, legacy
per-target names are accepted as aliases, and an environment variable overrides both. A band file
from the original campaign and a band file from a new target both work, and nothing in this
repository has to name a residue.

Logical terms:
    anchor_lo / anchor_hi   the acceptance window on the anchor-contact distance
    core_min                the minimum conserved-core contact count
    depth_band              the [min, max] insertion-depth window
    anchor_per_pose         the per-pose field holding the anchor distance

Override any of them per target, e.g.:
    export BAND_KEY_ANCHOR_LO=hbond_floor
    export BAND_KEY_ANCHOR_PER_POSE=hbond
"""
import os

# Generic names only. A scorer that names its anchor term after the target's own residue - which
# is the natural thing to do, and what the original campaign did - is handled by the environment
# override rather than by an alias list here, so this repository names no target. Example, for a
# band file whose keys are <residue>_subfloor / <residue>_max and whose per-pose field is
# <residue>:
#     export BAND_KEY_ANCHOR_LO=<residue>_subfloor
#     export BAND_KEY_ANCHOR_HI=<residue>_max
#     export BAND_KEY_ANCHOR_PER_POSE=<residue>
CANDIDATES = {
    "anchor_lo":       ("anchor_subfloor", "anchor_lo"),
    "anchor_hi":       ("anchor_max", "anchor_hi"),
    "core_min":        ("core_min",),
    "depth_band":      ("depth_band", "depth_band_Z"),
    "anchor_per_pose": ("anchor",),
}


def key(logical, mapping):
    """The actual key for `logical` inside `mapping` (a band dict or a per-pose record).

    Resolution order: $BAND_KEY_<LOGICAL> if set, then the candidate list. Raises with the
    candidates named, because a KeyError on a band file is otherwise very hard to read.
    """
    env = os.environ.get("BAND_KEY_" + logical.upper())
    if env:
        if env not in mapping:
            raise KeyError("%s: $BAND_KEY_%s is %r, which is not in %s"
                           % (logical, logical.upper(), env, sorted(mapping)))
        return env
    for c in CANDIDATES[logical]:
        if c in mapping:
            return c
    raise KeyError("%s: none of %s found in %s; set $BAND_KEY_%s to the right name"
                   % (logical, CANDIDATES[logical], sorted(mapping), logical.upper()))


def get(logical, mapping):
    """The value for `logical` out of `mapping`."""
    return mapping[key(logical, mapping)]


def band_terms(band):
    """(anchor_lo, anchor_hi, core_min, depth_band) out of a band dict, in one call.

    Every consumer of a band should go through this, so there is one implementation of what the
    band's terms are and not one per script.
    """
    return (get("anchor_lo", band), get("anchor_hi", band),
            get("core_min", band), get("depth_band", band))


def holds(sample, band):
    """Does one pose satisfy the WHOLE band?

    The conjunction is the method: a pose that satisfies two of three terms is not a hit, and
    scoring each term separately is how a design with no single good pose comes to look acceptable.
    """
    lo, hi, cmin, dz = band_terms(band)
    a = sample[key("anchor_per_pose", sample)]
    return lo <= a <= hi and sample["core"] >= cmin and dz[0] <= sample["depth_heavy_Z"] <= dz[1]
