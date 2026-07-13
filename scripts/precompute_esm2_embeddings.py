"""Fetch UniProt sequences for SL-universe genes and embed with ESM2.

Run once on a network node:
    uv run python scripts/precompute_esm2_embeddings.py \
        --benchmark-csv \
        data/SL_benchmark/derived/k562_depmap_rand_1to1/\
all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv \
        --out data/esm2/k562_sl_universe_esm2_650M.npz \
        --seq-cache data/esm2/symbol_to_sequence.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

logger = logging.getLogger("precompute_esm2")
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def symbols_from_csv(csv_path: Path, symbol_columns: tuple[str, ...]) -> list[str]:
    """Return sorted unique upper-case symbols from selected CSV columns."""
    frame = pd.read_csv(csv_path, usecols=list(symbol_columns))
    symbols: set[str] = set()
    for column in symbol_columns:
        symbols.update(frame[column].dropna().astype(str).str.upper())
    return sorted(symbols)


def universe_symbols(benchmark_csv: Path) -> list[str]:
    """Return sorted upper-case gene symbols from both columns of the benchmark CSV.

    Args:
        benchmark_csv: Path to the SL benchmark CSV with ``gene_a_symbol``
            and ``gene_b_symbol`` columns.

    Returns:
        Sorted list of unique upper-case gene symbols.
    """
    return symbols_from_csv(benchmark_csv, ("gene_a_symbol", "gene_b_symbol"))


def fetch_sequence(symbol: str) -> str | None:
    """Return the canonical human protein sequence for a gene symbol, or None.

    Queries UniProt REST for the top reviewed human hit. On any network
    error, logs a warning and returns ``None``.

    Args:
        symbol: Upper-case HGNC gene symbol.

    Returns:
        Amino-acid sequence string, or ``None`` if not found or on error.
    """
    query = f"(gene:{symbol}) AND (organism_id:9606) AND (reviewed:true)"
    params = urllib.parse.urlencode({"query": query, "format": "fasta", "size": 1})
    url = f"{UNIPROT_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - network best-effort, logged
        logger.warning("fetch failed for %s: %s", symbol, exc)
        return None
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines) or None


def load_or_fetch_sequences(symbols: list[str], cache: Path) -> dict[str, str]:
    """Load cached symbol→sequence map; fetch missing symbols from UniProt.

    Writes incrementally to ``cache`` every 100 new symbols.

    Args:
        symbols: Upper-case gene symbols to resolve.
        cache: Path to the JSON cache file (created if absent).

    Returns:
        Mapping from upper-case symbol to amino-acid sequence.
    """
    seqs: dict[str, str] = {}
    if cache.exists():
        try:
            seqs = json.loads(cache.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt JSON cache at %s; starting fresh", cache)
            seqs = {}
    for i, symbol in enumerate(symbols):
        if symbol in seqs:
            continue
        seq = fetch_sequence(symbol)
        if seq:
            seqs[symbol] = seq
            if len(seqs) % 100 == 0:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(seqs))
                logger.info("resolved %d/%d sequences", len(seqs), len(symbols))
        time.sleep(0.1)  # be polite to UniProt
    cache.write_text(json.dumps(seqs))
    return seqs


def truncate_sequence(seq: str, symbol: str, max_len: int = 1022) -> str:
    """Truncate a protein sequence to ``max_len`` tokens; warns when truncation occurs.

    Args:
        seq: Amino-acid sequence string.
        symbol: Gene symbol for log messages.
        max_len: Maximum allowed sequence length (default: 1022, ESM2 limit).

    Returns:
        The original sequence if within ``max_len``, otherwise the first
        ``max_len`` characters.
    """
    if len(seq) > max_len:
        logger.warning(
            "truncating %s from %d to %d residues for ESM2 input",
            symbol,
            len(seq),
            max_len,
        )
        return seq[:max_len]
    return seq


def check_resolution(resolved: np.ndarray, n_symbols: int) -> None:
    """Validate that a sufficient fraction of symbols were resolved to sequences.

    Args:
        resolved: Boolean array indicating which symbols have embeddings.
        n_symbols: Total number of symbols in the universe.

    Raises:
        RuntimeError: If no symbols resolved (all-zero ``resolved`` array).
    """
    n_resolved = int(resolved.sum())
    if n_resolved == 0:
        raise RuntimeError("no sequences resolved; aborting embedding write")
    frac = n_resolved / n_symbols if n_symbols > 0 else 0.0
    if frac < 0.5:
        logger.warning(
            "low resolution: only %d/%d symbols resolved (%.1f%%)",
            n_resolved,
            n_symbols,
            frac * 100,
        )


def mean_pool_residues(
    hidden: np.ndarray, special_tokens_mask: np.ndarray
) -> np.ndarray:
    """Mean-pool token embeddings over residues, excluding special tokens.

    ESM2 prepends a BOS (``<cls>``) and appends an EOS (``<eos>``) token; padding
    tokens may also be present. The spec calls for a residue mean, so special
    tokens (``special_tokens_mask == 1``) are excluded from the average.

    Args:
        hidden: Token embeddings, shape ``(n_tokens, dim)``.
        special_tokens_mask: Per-token mask, shape ``(n_tokens,)``; ``1`` marks a
            special (non-residue) token to exclude.

    Returns:
        Residue-mean embedding, shape ``(dim,)``. If every token is special
        (degenerate input), falls back to a full mean over all tokens to avoid
        a NaN.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    keep = np.asarray(special_tokens_mask) == 0
    if not keep.any():
        return hidden.mean(axis=0)
    return hidden[keep].mean(axis=0)


def embed_sequences(
    symbols: list[str],
    seqs: dict[str, str],
    model_name: str,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed resolved sequences with ESM2; unresolved rows stay zero.

    Args:
        symbols: Upper-case gene symbols in universe order.
        seqs: Mapping from symbol to amino-acid sequence.
        model_name: HuggingFace model ID, e.g. ``"facebook/esm2_t33_650M_UR50D"``.
        cache_dir: Optional Hugging Face cache directory for model/tokenizer files.
        local_files_only: If True, require the model/tokenizer to already be in
            the local Hugging Face cache.

    Returns:
        A tuple ``(vectors, resolved)`` where ``vectors`` has shape
        ``(n_gene, hidden_size)`` and ``resolved`` is a boolean array.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from_pretrained_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    from_pretrained_kwargs = {
        key: value for key, value in from_pretrained_kwargs.items() if value is not None
    }
    tokenizer = EsmTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)
    model = (
        EsmModel.from_pretrained(model_name, **from_pretrained_kwargs).to(device).eval()
    )
    dim = model.config.hidden_size
    vectors = np.zeros((len(symbols), dim), dtype=np.float32)
    resolved = np.zeros(len(symbols), dtype=bool)
    with torch.no_grad():
        for row, symbol in enumerate(symbols):
            seq = seqs.get(symbol)
            if not seq:
                continue
            toks = tokenizer(
                truncate_sequence(seq, symbol),
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = toks.pop("special_tokens_mask")[0].cpu().numpy()
            toks = toks.to(device)
            out = model(**toks).last_hidden_state[0]  # (L, dim)
            vectors[row] = mean_pool_residues(out.cpu().numpy(), special)
            resolved[row] = True
            if row % 200 == 0:
                logger.info("embedded %d/%d", row, len(symbols))
    return vectors, resolved


def main() -> None:
    """CLI entry point: fetch sequences and embed with ESM2."""
    parser = argparse.ArgumentParser(
        description="Fetch UniProt sequences and embed with ESM2."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        required=True,
        help="SL benchmark CSV with gene_a_symbol and gene_b_symbol columns.",
    )
    parser.add_argument(
        "--symbol-column",
        action="append",
        default=None,
        help=(
            "CSV symbol column; repeat for multiple columns. Defaults to the "
            "exp08 gene_a_symbol and gene_b_symbol columns."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npz path.",
    )
    parser.add_argument(
        "--seq-cache",
        type=Path,
        required=True,
        help="JSON cache for symbol→sequence (incremental, reuse across runs).",
    )
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace ESM2 model ID.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for model/tokenizer downloads.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-cached Hugging Face files; do not download.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    symbol_columns = (
        tuple(args.symbol_column)
        if args.symbol_column
        else ("gene_a_symbol", "gene_b_symbol")
    )
    symbols = symbols_from_csv(args.benchmark_csv, symbol_columns)
    logger.info("universe size: %d genes", len(symbols))
    seqs = load_or_fetch_sequences(symbols, args.seq_cache)
    vectors, resolved = embed_sequences(
        symbols,
        seqs,
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    check_resolution(resolved, n_symbols=len(symbols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        symbols=np.array(symbols, dtype=object),
        vectors=vectors,
        resolved=resolved,
    )
    logger.info(
        "wrote %s (%d resolved / %d)", args.out, int(resolved.sum()), len(symbols)
    )


if __name__ == "__main__":
    main()
