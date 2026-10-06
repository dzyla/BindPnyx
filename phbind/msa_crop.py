"""Crop an a3m by MATCH columns (1-based, inclusive). Lowercase = insertions: kept only between kept match columns.
Used to drop the 5 disordered N-terminal residues (local 1-5) that the organiser PDB lacks, so the MSA query equals the shard sequence."""
import sys

def crop_a3m(src, dst, first, last):
    out, n_rows = [], 0
    hdr = None
    for line in open(src, "rb").read().decode().split("\n"):
        if line.startswith(">"):
            hdr = line; continue
        if not line or hdr is None: continue
        col, keep = 0, []
        for ch in line:
            if ch.islower():
                if first < col < last: keep.append(ch)          # insertion strictly inside the window
                continue
            col += 1
            if first <= col <= last: keep.append(ch)
        assert col >= last, f"row shorter than window ({col} < {last} match columns): {hdr[:40]}"
        out += [hdr, "".join(keep)]; n_rows += 1; hdr = None
    open(dst, "w").write("\n".join(out) + "\n")
    return n_rows

if __name__ == "__main__":
    print(crop_a3m(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])), "rows")
