"""Fetch gf-12L-30M-i2048 and the gc95M and gc30M dictionaries of Geneformer.

The files are those of Hugging Face ctheodoris/Geneformer at revision 01d3ea8
(2025-01-10), the last one that still carries gf-12L-30M-i2048, the model the released
Cilantro-SL code loads. The H20 cannot download them from Hugging Face (hf-mirror
redirects large files to Xet storage, which it cannot resolve), so they come from
ModelScope's copy, ZhejiangLab-LifeScience/Geneformer, and each is checked against the
SHA-256 (LFS files) or git blob SHA-1 that Hugging Face reports for that revision. The
gc95M gene-name dictionary differs there and is not fetched: no step loads it.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GENEFORMER_DATA  # noqa: E402

REVISION = "01d3ea8993c2a49704375d7ee6175210cfa33f11"
BASE = "https://www.modelscope.cn/models/ZhejiangLab-LifeScience/Geneformer/resolve/master/"
FILES = {
    "gf-12L-30M-i2048/config.json": (
        "git-sha1",
        "52a12424cea85facdf0ca0c507908506daae7ea7",
    ),
    "gf-12L-30M-i2048/pytorch_model.bin": (
        "sha256",
        "812f8d85e5ecf9d64c268f052f6ece2c1906bc4f1aecf70d5144b2598386b615",
    ),
    "gf-12L-30M-i2048/training_args.bin": (
        "sha256",
        "259cf6067211e24e198690d00f0a222ee5550ad57e23d04ced0d0ca2e1b3738e",
    ),
    "geneformer/token_dictionary_gc95M.pkl": (
        "sha256",
        "67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f",
    ),
    "geneformer/gene_median_dictionary_gc95M.pkl": (
        "sha256",
        "a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5",
    ),
    "geneformer/ensembl_mapping_dict_gc95M.pkl": (
        "sha256",
        "0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c",
    ),
    "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl": (
        "sha256",
        "ab9dc40973fa5224d77b793e2fd114cacf3d08423ed9c4c49caf0ba9c7f218f1",
    ),
    "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl": (
        "sha256",
        "b3b589bb5ec75040d05fc44dd6bf0184cf87f3c362cf158d196a6ed3b7fe5f39",
    ),
    "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl": (
        "sha256",
        "55e67962e79c0039a6c32d43c5c99f38e51964bbcfa32f736150ee1e285c438c",
    ),
    "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl": (
        "sha256",
        "eac0fb0b3007267871b6305ac0003ceba19d4f28d85686cb9067ecf142787869",
    ),
}


def digest(data: bytes, kind: str) -> str:
    if kind == "git-sha1":
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    for name, (kind, expected) in FILES.items():
        target = GENEFORMER_DATA / name
        if target.exists() and digest(target.read_bytes(), kind) == expected:
            print(f"ok      {name}")
            continue
        request = urllib.request.Request(
            BASE + name, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = response.read()
        actual = digest(data, kind)
        if actual != expected:
            raise SystemExit(f"{name}: {kind} {actual} != {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)
        print(f"fetched {name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
