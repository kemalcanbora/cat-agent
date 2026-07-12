"""Benchmark PDF extraction before considering a native parser replacement.

With no path, the script creates a text-heavy synthetic PDF using matplotlib.
Pass a representative production PDF for a meaningful fidelity/performance
decision:

    python benchmarks/benchmark_pdf_parser.py contract.pdf --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from cat_agent.tools.parsers.pdf_parser import parse_pdf, parse_pdf_native


def _synthetic_pdf(path: Path, pages: int) -> None:
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Generating a synthetic PDF requires matplotlib; pass an existing PDF path instead."
        ) from error

    paragraph = (
        "Cat-Agent document ingestion benchmark. This page contains repeated "
        "text blocks for PDF layout extraction and paragraph merging."
    )
    with PdfPages(path) as pdf:
        for page_number in range(1, pages + 1):
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(0.08, 0.94, f"Benchmark page {page_number}", fontsize=16)
            for row in range(20):
                figure.text(0.08, 0.89 - row * 0.04, paragraph, fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path)
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    temporary = None
    pdf_path = args.pdf
    if pdf_path is None:
        temporary = tempfile.TemporaryDirectory()
        pdf_path = Path(temporary.name) / "synthetic.pdf"
        _synthetic_pdf(pdf_path, args.pages)

    samples = []
    parsed = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        parsed = parse_pdf(str(pdf_path))
        samples.append((time.perf_counter() - started) * 1000)

    page_count = len(parsed or [])
    item_count = sum(len(page["content"]) for page in (parsed or []))
    print(f"PDF: {pdf_path}")
    print(f"Pages: {page_count}; extracted items: {item_count}")
    print(
        f"Parse time: mean={statistics.mean(samples):.2f} ms "
        f"median={statistics.median(samples):.2f} ms"
    )
    if page_count:
        print(f"Mean per page: {statistics.mean(samples) / page_count:.2f} ms")

    try:
        native_parsed = parse_pdf_native(str(pdf_path))
    except ImportError:
        print("Rust PDF prototype: unavailable (build the native extension)")
    else:
        native_samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            native_parsed = parse_pdf_native(str(pdf_path))
            native_samples.append((time.perf_counter() - started) * 1000)
        native_items = sum(len(page["content"]) for page in native_parsed)
        print(f"Rust extracted pages: {len(native_parsed)}; items: {native_items}")
        print(
            f"Rust parse time: mean={statistics.mean(native_samples):.2f} ms "
            f"median={statistics.median(native_samples):.2f} ms"
        )

    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
