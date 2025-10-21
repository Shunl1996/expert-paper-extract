#!/usr/bin/env python3
"""
PDF word and token counter.

Features:
- Scans given files/directories for .pdf files (default: current directory).
- Extracts text using available backends (auto-detect):
  - tiktoken (optional) for exact token count if installed.
  - pypdf or PyPDF2 (if installed) or pdfminer.six (if installed).
  - As fallback, tries the `pdftotext` command if present on PATH.
- Reports pages, characters, words, and token estimates.
- Can export results to CSV.

Usage examples:
  python check_length.py                          # scan current directory
  python check_length.py some.pdf other_dir       # scan specific targets
  python check_length.py --csv summary.csv        # also write CSV
  python check_length.py --engine pdfminer        # force backend

Notes on tokens:
- If `tiktoken` is available, counts tokens with `cl100k_base` encoding.
- Otherwise, provides two rough estimates:
    tokens≈words/0.75 (OpenAI heuristic: ~75 words ≈ 100 tokens)
    tokens≈characters/4 (OpenAI heuristic: ~1 token ≈ 4 characters)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# -----------------------------
# Optional tokenizer (tiktoken)
# -----------------------------
_tiktoken = None
try:
    import tiktoken  # type: ignore

    _tiktoken = tiktoken
except Exception:
    _tiktoken = None


# -----------------------------
# PDF extraction backends
# -----------------------------
def _extract_with_pypdf(path: Path) -> Tuple[str, int]:
    """Extract text using pypdf or PyPDF2. Returns (text, pages)."""
    try:
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            pages = len(reader.pages)
            text_parts = []
            for p in reader.pages:
                try:
                    text_parts.append(p.extract_text() or "")
                except Exception:
                    text_parts.append("")
            return ("\n".join(text_parts), pages)
        except Exception:
            import PyPDF2  # type: ignore

            reader = PyPDF2.PdfReader(str(path))
            pages = len(reader.pages)
            text_parts = []
            for p in reader.pages:
                try:
                    text_parts.append(p.extract_text() or "")
                except Exception:
                    text_parts.append("")
            return ("\n".join(text_parts), pages)
    except Exception as e:
        raise RuntimeError(f"pypdf/PyPDF2 failed: {e}")


def _extract_with_pdfminer(path: Path) -> Tuple[str, int]:
    """Extract text using pdfminer.six. Returns (text, pages)."""
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        from pdfminer.pdfpage import PDFPage  # type: ignore

        text = extract_text(str(path)) or ""
        # Page count via PDFPage
        with open(path, "rb") as f:
            pages = sum(1 for _ in PDFPage.get_pages(f))
        return (text, pages)
    except Exception as e:
        raise RuntimeError(f"pdfminer.six failed: {e}")


def _extract_with_pdftotext(path: Path) -> Tuple[str, int]:
    """Extract text using the system `pdftotext` (poppler). Returns (text, pages)."""
    if not shutil.which("pdftotext"):
        raise RuntimeError("`pdftotext` not found in PATH")
    # Get page count using pdfinfo if present
    pages = 0
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        try:
            out = subprocess.check_output([pdfinfo, str(path)], stderr=subprocess.DEVNULL)
            for line in out.decode("utf-8", errors="ignore").splitlines():
                if line.lower().startswith("pages:"):
                    try:
                        pages = int(line.split(":", 1)[1].strip())
                    except Exception:
                        pages = 0
                    break
        except Exception:
            pages = 0

    try:
        out = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], stderr=subprocess.DEVNULL)
        text = out.decode("utf-8", errors="ignore")
        return (text, pages)
    except Exception as e:
        raise RuntimeError(f"pdftotext failed: {e}")


def extract_text_auto(path: Path, engine: str = "auto") -> Tuple[str, int, str]:
    """Extract text and return (text, pages, engine_used)."""
    engines = []
    if engine == "auto":
        engines = ["pypdf", "pdfminer", "pdftotext"]
    else:
        engines = [engine]

    last_err = None
    for eng in engines:
        try:
            if eng == "pypdf":
                text, pages = _extract_with_pypdf(path)
            elif eng == "pdfminer":
                text, pages = _extract_with_pdfminer(path)
            elif eng == "pdftotext":
                text, pages = _extract_with_pdftotext(path)
            else:
                raise ValueError(f"Unknown engine: {eng}")
            return text, pages, eng
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All text extraction engines failed for {path.name}: {last_err}")


# -----------------------------
# Counting helpers
# -----------------------------
_WORD_RE = re.compile(r"\b\w+[\w'-]*\b", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize whitespace and de-hyphenate common line breaks."""
    # Merge hyphenated line breaks like "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace newlines and tabs with spaces and collapse multiple spaces
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def estimate_tokens(words: int, chars: int) -> Tuple[int, int]:
    """Return (tokens_by_words, tokens_by_chars) using heuristics.

    - tokens_by_words ≈ words / 0.75 (OpenAI heuristic)
    - tokens_by_chars ≈ chars / 4
    """
    tw = int(round(words / 0.75)) if words > 0 else 0
    tc = int(round(chars / 4)) if chars > 0 else 0
    return tw, tc


def tiktoken_count(text: str) -> Optional[int]:
    if _tiktoken is None:
        return None
    try:
        enc = _tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            enc = _tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            return None
    try:
        return len(enc.encode(text))
    except Exception:
        return None


# -----------------------------
# Scanning and CLI
# -----------------------------
@dataclass
class PdfStats:
    path: Path
    pages: int
    chars: int
    words: int
    tokens_word_est: int
    tokens_char_est: int
    tokens_tiktoken: Optional[int]
    engine: str
    # Image-related (optional)
    image_mp_total: Optional[float] = None
    image_tokens_est: Optional[int] = None
    # Totals combining text + image (optional)
    total_tokens_word_est: Optional[int] = None
    total_tokens_char_est: Optional[int] = None
    total_tokens_tiktoken: Optional[int] = None


def iter_pdfs(targets: List[Path]) -> Iterable[Path]:
    for t in targets:
        if t.is_file() and t.suffix.lower() == ".pdf":
            yield t
        elif t.is_dir():
            for root, _dirs, files in os.walk(t):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        yield Path(root) / f


def analyze_pdf(path: Path, engine: str = "auto") -> PdfStats:
    text, pages, engine_used = extract_text_auto(path, engine=engine)
    norm = normalize_text(text)
    chars = len(norm)
    words = count_words(norm)
    est_w, est_c = estimate_tokens(words, chars)
    tk = tiktoken_count(norm)
    return PdfStats(
        path=path,
        pages=pages,
        chars=chars,
        words=words,
        tokens_word_est=est_w,
        tokens_char_est=est_c,
        tokens_tiktoken=tk,
        engine=engine_used,
    )


def format_row(stats: PdfStats) -> str:
    tk_display = (
        str(stats.tokens_tiktoken) if stats.tokens_tiktoken is not None else "-"
    )
    parts = [
        f"{stats.path.name}\tpages={stats.pages}\tchars={stats.chars}\t"
        f"words={stats.words}\ttokens(word-est)={stats.tokens_word_est}\t"
        f"tokens(char-est)={stats.tokens_char_est}\ttokens(tiktoken)={tk_display}\t"
        f"engine={stats.engine}"
    ]
    if stats.image_mp_total is not None:
        parts.append(f"image_mp={stats.image_mp_total:.2f}")
    if stats.image_tokens_est is not None:
        parts.append(f"image_tokens={stats.image_tokens_est}")
    if stats.total_tokens_word_est is not None:
        parts.append(f"total_tokens(word-est)={stats.total_tokens_word_est}")
    if stats.total_tokens_char_est is not None:
        parts.append(f"total_tokens(char-est)={stats.total_tokens_char_est}")
    if stats.total_tokens_tiktoken is not None:
        parts.append(f"total_tokens(tiktoken)={stats.total_tokens_tiktoken}")
    return "\t".join(parts)


def write_csv(rows: List[PdfStats], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "pages",
                "characters",
                "words",
                "tokens_word_est",
                "tokens_char_est",
                "tokens_tiktoken",
                "engine",
                "image_mp",
                "image_tokens",
                "total_tokens_word_est",
                "total_tokens_char_est",
                "total_tokens_tiktoken",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path.name,
                    r.pages,
                    r.chars,
                    r.words,
                    r.tokens_word_est,
                    r.tokens_char_est,
                    r.tokens_tiktoken if r.tokens_tiktoken is not None else "",
                    r.engine,
                    f"{r.image_mp_total:.4f}" if r.image_mp_total is not None else "",
                    r.image_tokens_est if r.image_tokens_est is not None else "",
                    r.total_tokens_word_est if r.total_tokens_word_est is not None else "",
                    r.total_tokens_char_est if r.total_tokens_char_est is not None else "",
                    r.total_tokens_tiktoken if r.total_tokens_tiktoken is not None else "",
                ]
            )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count words and estimate tokens for PDFs.")
    p.add_argument(
        "targets",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Files or directories to scan (default: current directory)",
    )
    p.add_argument(
        "--engine",
        choices=["auto", "pypdf", "pdfminer", "pdftotext"],
        default="auto",
        help="Force a specific text extraction backend.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write a CSV summary.",
    )
    # Multimodal image estimation options
    p.add_argument(
        "--image",
        action="store_true",
        help=(
            "Estimate image megapixels per page and image tokens if a rate is provided."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "DPI used to estimate pixel dimensions from page size (default: 150)."
        ),
    )
    p.add_argument(
        "--image-tokens-per-mp",
        type=float,
        default=None,
        help=(
            "Tokens per megapixel for your target model (provider-specific)."
            " If set, total image tokens are estimated as MP * this rate."
        ),
    )
    p.add_argument(
        "--image-divisor",
        type=float,
        default=None,
        help=(
            "Alternative formula: tokens = total_pixels / DIVISOR (e.g., 750)."
            " If provided, takes precedence over --image-tokens-per-mp."
        ),
    )
    p.add_argument(
        "--default-page-size",
        choices=["letter", "a4"],
        default="letter",
        help=(
            "Fallback page size when exact PDF sizes are unavailable (letter=8.5x11in, a4=8.27x11.69in)."
        ),
    )
    p.add_argument(
        "--default-page-width-in",
        type=float,
        default=None,
        help="Override default page width in inches (takes precedence over --default-page-size).",
    )
    p.add_argument(
        "--default-page-height-in",
        type=float,
        default=None,
        help="Override default page height in inches (takes precedence over --default-page-size).",
    )
    return p.parse_args(argv)


def _get_pdf_page_sizes_in_inches(path: Path) -> Optional[List[Tuple[float, float]]]:
    """Return list of (width_in, height_in) per page if we can read via pypdf/PyPDF2.

    Returns None on failure (we'll fall back to defaults).
    """
    try:
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            sizes = []
            for p in reader.pages:
                box = p.mediabox
                # Points to inches (72 points per inch)
                w_pt = float(box.right) - float(box.left)
                h_pt = float(box.top) - float(box.bottom)
                sizes.append((w_pt / 72.0, h_pt / 72.0))
            return sizes
        except Exception:
            import PyPDF2  # type: ignore

            reader = PyPDF2.PdfReader(str(path))
            sizes = []
            for p in reader.pages:
                box = p.mediabox
                w_pt = float(box.right) - float(box.left)
                h_pt = float(box.top) - float(box.bottom)
                sizes.append((w_pt / 72.0, h_pt / 72.0))
            return sizes
    except Exception:
        return None


def _default_page_size_inches(default_page_size: str, w_override: Optional[float], h_override: Optional[float]) -> Tuple[float, float]:
    if w_override is not None and h_override is not None:
        return (w_override, h_override)
    if default_page_size == "a4":
        return (8.27, 11.69)
    # letter default
    return (8.5, 11.0)


def estimate_image_megapixels(path: Path, dpi: int, default_page_size: str, w_override: Optional[float], h_override: Optional[float], fallback_pages: Optional[int] = None) -> Tuple[float, int]:
    """Estimate total megapixels for all pages, given DPI and page sizes.

    Returns (mp_total, page_count_used).
    """
    sizes = _get_pdf_page_sizes_in_inches(path)
    if not sizes:
        default_w, default_h = _default_page_size_inches(default_page_size, w_override, h_override)
        # Try to get page count via pypdf/PyPDF2; otherwise assume 1
        page_count = fallback_pages if fallback_pages is not None else 1
        if page_count is None or page_count <= 0:
            page_count = 1
        if page_count == 1:
            # As a last resort, try pypdf/PyPDF2 for page count
            try:
                try:
                    import pypdf  # type: ignore

                    reader = pypdf.PdfReader(str(path))
                    page_count = len(reader.pages)
                except Exception:
                    import PyPDF2  # type: ignore

                    reader = PyPDF2.PdfReader(str(path))
                    page_count = len(reader.pages)
            except Exception:
                pass
        # Compute MP with defaults
        pixels_per_page = (default_w * dpi) * (default_h * dpi)
        mp_per_page = pixels_per_page / 1_000_000.0
        return (mp_per_page * page_count, page_count)

    # Use actual sizes
    mp_total = 0.0
    for (w_in, h_in) in sizes:
        pixels = (w_in * dpi) * (h_in * dpi)
        mp_total += pixels / 1_000_000.0
    return (mp_total, len(sizes))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    pdfs = sorted(set(iter_pdfs([p.resolve() for p in args.targets])))
    if not pdfs:
        print("No PDF files found.")
        return 1

    results: List[PdfStats] = []
    for pdf in pdfs:
        try:
            stats = analyze_pdf(pdf, engine=args.engine)
            # Image estimation if requested
            if args.image or args.image_tokens_per_mp is not None or args.image_divisor is not None:
                mp_total, _ = estimate_image_megapixels(
                    pdf,
                    dpi=args.dpi,
                    default_page_size=args.default_page_size,
                    w_override=args.default_page_width_in,
                    h_override=args.default_page_height_in,
                    fallback_pages=stats.pages,
                )
                stats.image_mp_total = mp_total
                if args.image_divisor is not None:
                    total_pixels = mp_total * 1_000_000.0
                    stats.image_tokens_est = int(round(total_pixels / args.image_divisor))
                elif args.image_tokens_per_mp is not None:
                    stats.image_tokens_est = int(round(mp_total * args.image_tokens_per_mp))
                # Combine totals whenever we have image tokens
                if stats.image_tokens_est is not None:
                    stats.total_tokens_word_est = stats.tokens_word_est + stats.image_tokens_est
                    stats.total_tokens_char_est = stats.tokens_char_est + stats.image_tokens_est
                    if stats.tokens_tiktoken is not None:
                        stats.total_tokens_tiktoken = stats.tokens_tiktoken + stats.image_tokens_est
            results.append(stats)
            print(format_row(stats))
        except Exception as e:
            print(f"ERROR\t{pdf.name}\t{e}")

    if args.csv:
        try:
            write_csv(results, args.csv)
            print(f"Wrote CSV: {args.csv}")
        except Exception as e:
            print(f"Failed to write CSV to {args.csv}: {e}")

    # Print a tiny total summary
    total_words = sum(r.words for r in results)
    total_chars = sum(r.chars for r in results)
    est_w, est_c = estimate_tokens(total_words, total_chars)
    tk_sum = (
        sum(r.tokens_tiktoken for r in results if r.tokens_tiktoken is not None)
        if any(r.tokens_tiktoken is not None for r in results)
        else None
    )
    tk_display = str(tk_sum) if tk_sum is not None else "-"
    # Image totals (optional)
    image_mp_total_sum = (
        sum(r.image_mp_total for r in results if r.image_mp_total is not None)
        if any(r.image_mp_total is not None for r in results)
        else None
    )
    image_tokens_sum = (
        sum(r.image_tokens_est for r in results if r.image_tokens_est is not None)
        if any(r.image_tokens_est is not None for r in results)
        else None
    )
    parts = [
        f"TOTAL\tfiles={len(results)}\twords={total_words}\tchars={total_chars}",
        f"tokens(word-est)={est_w}",
        f"tokens(char-est)={est_c}",
        f"tokens(tiktoken)={tk_display}",
    ]
    if image_mp_total_sum is not None:
        parts.append(f"image_mp={image_mp_total_sum:.2f}")
    if image_tokens_sum is not None:
        parts.append(f"image_tokens={image_tokens_sum}")
        parts.append(f"total_tokens(word-est)={est_w + image_tokens_sum}")
        parts.append(f"total_tokens(char-est)={est_c + image_tokens_sum}")
        if tk_sum is not None:
            parts.append(f"total_tokens(tiktoken)={tk_sum + image_tokens_sum}")
    print("\t".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
