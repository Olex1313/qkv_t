"""Parse MNN attention benchmark logs and plot GFLOPS vs seq_len.

One SVG is generated per unique (B, H, D) combination found across all logs.
Series are backend/tag (e.g. opencl/flash, vulkan/native).
Multiple files with the same backend are averaged.

Usage:
    python plot_mnn_bench.py results.opencl.log results.vulkan.log
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# tag=flash B=4  seq=1024  H= 4  D= 64 min=... avg=... max=... us  GFLOPS=1.2345
_LINE_RE = re.compile(
    r"tag=(\S+)\s+B=(\d+)\s+seq=\s*(\d+)\s+H=\s*(\d+)\s+D=\s*(\d+)"
    r".*?min=\s*([\d.]+).*?avg=\s*([\d.]+).*?max=\s*([\d.]+)\s+us\s+GFLOPS=([\d.]+)"
)


def parse_log(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        m = _LINE_RE.search(line)
        if m:
            tag, B, seq, H, D, mn, avg, mx, gflops = m.groups()
            records.append(dict(
                tag=tag, B=int(B), seq=int(seq), H=int(H), D=int(D),
                min_us=float(mn), avg_us=float(avg), max_us=float(mx),
                gflops=float(gflops),
            ))
    return records


def plot_all(log_paths: list[Path], out_dir: Path = Path(".")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # data[(B,H,D)][series_label][seq] = [gflops, ...]
    data: dict[tuple, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for path in log_paths:
        parts = path.name.split(".")
        backend = parts[-2] if len(parts) >= 3 else path.stem
        records = parse_log(path)
        if not records:
            print(f"Warning: no results parsed from {path}")
            continue
        for r in records:
            key = (r["B"], r["H"], r["D"])
            label = f"{backend}/{r['tag']}"
            data[key][label][r["seq"]].append(r["gflops"])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (B, H, D), series in sorted(data.items()):
        # average repeated runs
        avg: dict[str, dict[int, float]] = {
            label: {seq: sum(vals) / len(vals) for seq, vals in seqs.items()}
            for label, seqs in series.items()
        }

        seq_lens = sorted({s for v in avg.values() for s in v})
        names = list(avg.keys())
        positions = np.arange(len(seq_lens))
        width = 0.7 / len(names)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(names):
            gflops = [avg[name].get(s, 0) for s in seq_lens]
            pos = positions + (i - (len(names) - 1) / 2) * width
            ax.bar(pos, gflops, width=width * 0.85,
                   color=colors[i % len(colors)], alpha=0.75, label=name)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(s) for s in seq_lens])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("GFLOPS")
        ax.set_title(f"MNN Attention  B={B}  H={H}  D={D}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        svg_path = out_dir / f"bench_mnn_B{B}_H{H}_D{D}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Saved {svg_path}")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mnn_bench.py file1.backend.log [file2.backend.log ...]")
        sys.exit(1)
    plot_all([Path(p) for p in sys.argv[1:]])
