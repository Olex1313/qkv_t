"""Parse bench_kernels log and plot GFLOPS vs seq_len.

One SVG per unique (B, H, D) combination.
Series are the tag (vk-flash, vk-coopmat, ocl-flash, etc.).

Usage:
    python plot_bench.py bench.log [bench2.log ...]
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# vk-flash              B=1 H= 4 L= 1024 D= 64  min=... avg=... max=... us  GFLOPS=964.15
_LINE_RE = re.compile(
    r"^(\S+)\s+B=(\d+)\s+H=\s*(\d+)\s+L=\s*(\d+)\s+D=\s*(\d+)"
    r".*?min=\s*([\d.]+).*?avg=\s*([\d.]+).*?max=\s*([\d.]+)\s+us\s+GFLOPS=([\d.]+)"
)


def parse_log(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        m = _LINE_RE.search(line)
        if m:
            tag, B, H, L, D, mn, avg, mx, gflops = m.groups()
            records.append(dict(
                tag=tag, B=int(B), H=int(H), L=int(L), D=int(D),
                min_us=float(mn), avg_us=float(avg), max_us=float(mx),
                gflops=float(gflops),
            ))
    return records


def plot_all(log_paths: list[Path], out_dir: Path = Path(".")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # data[(B,H,D)][tag][L] = [gflops, ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for path in log_paths:
        for r in parse_log(path):
            data[(r["B"], r["H"], r["D"])][r["tag"]][r["L"]].append(r["gflops"])

    if not data:
        print("No results parsed.")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (B, H, D), series in sorted(data.items()):
        avg = {
            tag: {L: sum(v)/len(v) for L, v in seqs.items()}
            for tag, seqs in series.items()
        }

        seq_lens = sorted({L for v in avg.values() for L in v})
        names = sorted(avg.keys())
        positions = np.arange(len(seq_lens))
        width = 0.7 / len(names)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(names):
            gflops = [avg[name].get(L, 0) for L in seq_lens]
            pos = positions + (i - (len(names) - 1) / 2) * width
            ax.bar(pos, gflops, width=width * 0.85,
                   color=colors[i % len(colors)], alpha=0.75, label=name)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(L) for L in seq_lens], rotation=30, ha="right")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("GFLOPS")
        ax.set_title(f"Flash Attention  B={B}  H={H}  D={D}")
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        svg_path = out_dir / f"bench_B{B}_H{H}_D{D}.png"
        fig.savefig(svg_path, format="png", dpi=150, bbox_inches="tight")
        print(f"Saved {svg_path}")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_bench.py bench.log [bench2.log ...]")
        sys.exit(1)
    plot_all([Path(p) for p in sys.argv[1:]])
