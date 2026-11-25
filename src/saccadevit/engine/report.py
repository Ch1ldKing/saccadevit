"""Collect output/ into docs/CURRENT_RESULTS.md: run registry, test accuracy (mean ± std over
seeds), tau* operating points, latency, and hypothesis comparisons. Re-run any time."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT = Path("output")


def _ms(xs: list[float], pct: bool = True) -> str:
    if not xs:
        return "–"
    f = 100 if pct else 1
    m = np.mean(xs) * f
    return f"{m:.2f}" if len(xs) == 1 else f"{m:.2f} ± {np.std(xs) * f:.2f}"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _runs() -> dict[str, list[Path]]:
    runs = defaultdict(list)
    for cfg in sorted(OUT.glob("*/seed*/config.yaml")):
        runs[cfg.parent.parent.name].append(cfg.parent)
    return runs


def _status(d: Path, trains: bool) -> str:
    """✓ finished · ▶ running / partial · – not started"""
    if not d.exists():
        return "–"
    if trains:
        return "✓" if (d / "done").exists() else "▶"
    return "✓" if any(d.glob("summary_test.json")) or any(d.glob("sweep.csv")) else "▶"


def _registry(runs) -> str:
    from ..config import EXP_ROOT, load_config

    rows = []
    for p in sorted(EXP_ROOT.rglob("*.yaml")):
        c = load_config(p)
        m, name = c.model, c.name
        trains = not str(c.eval.ckpt).startswith("output/") or str(c.eval.ckpt).startswith(f"output/{name}/")
        what = m.kind if m.kind != "saccade" else f"{m.policy}, q={m.task}, K={m.k_max}, ρ={m.keep_ratio}"
        role = f"train ({c.train.stage})" if trains else f"eval of {Path(c.eval.ckpt).parts[1]}"
        seeds = " ".join(_status(OUT / name / f"seed{s}", trains) for s in range(3))
        val = [max(json.loads(l)["val_acc"] for l in open(d / "metrics.jsonl"))
               for d in runs.get(name, []) if (d / "metrics.jsonl").exists() and (d / "done").exists()]
        rows.append([name, c.data.dataset, what, role, seeds, _ms(val), f"`output/{name}/seed*/`"])
    return _table(["Experiment", "Data", "Model / policy", "Role", "Seeds 0 1 2", "Best val acc %", "Path"], rows)


def _accuracy(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        sums = [json.loads((d / "summary_test.json").read_text()) for d in dirs if (d / "summary_test.json").exists()]
        if not sums:
            continue
        ks = sorted({k for s in sums for k in s if k.startswith("acc@")}, key=lambda k: int(k[4:]))
        cells = {k: _ms([s[k] for s in sums if k in s]) for k in ks}
        npz = dirs[0] / "eval_test.npz"
        tok = int(np.load(npz)["tokens"][0]) if npz.exists() and len(np.load(npz)["tokens"]) else "–"
        row = [name, len(sums)] + [cells.get(f"acc@{i}", "–") for i in range(4)] + [tok]
        rows.append(row)
    return _table(["Experiment", "Seeds", "acc@0 (coarse / full)", "acc@1", "acc@2", "acc@3", "Tokens/look"], rows)


def _sweep(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        for d in dirs:
            p = d / "sweep.csv"
            if p.exists():
                for r in csv.DictReader(open(p)):
                    if r["rule"] == "tau*":
                        rows.append([name, d.name, r["keep_ratio"], r["tau"], f"{100 * float(r['acc']):.2f}",
                                     f"{float(r['looks']):.2f}", f"{float(r['rel_cost']):.3f}"])
    return _table(["Experiment", "Seed", "ρ", "τ*", "Test acc %", "Mean looks", "Rel. cost (token-blocks)"], rows) if rows else "_no sweeps yet_"


def _latency(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        for p in sorted(dirs[0].glob("latency_*.json")):
            r = json.loads(p.read_text())
            sec = ", ".join(f"{k} {v:.2f}" for k, v in r["sections_ms_per_img"].items())
            mem = f"{r['peak_mem_mb']:.0f}" if r["peak_mem_mb"] else "–"
            mode = ("graph" if r.get("graph") else "eager") + (f" b{r['batch']}" if r.get("batch", 1) > 1 else "")
            rows.append([name, r["device"], mode, r["keep_ratio"], r.get("k_max", "–"), r["tau"], f"{r['p50_ms']:.2f}",
                         f"{r['p95_ms']:.2f}", f"{r.get('img_per_s', 1e3 / r['p50_ms']):.0f}", f"{r['mean_looks']:.2f}",
                         f"{r['gflops_per_img']:.3f}", mem, sec])
    head = ["Experiment", "Device", "Mode", "ρ", "K", "τ", "p50 ms", "p95 ms", "img/s", "Looks", "GFLOPs", "Peak MB", "Per-module ms (eager)"]
    return _table(head, rows) if rows else "_no latency runs yet_"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pareto(prefix: str) -> str:
    """T2-T4: accuracy (test, mean over seeds) vs batch-1 GPU ms with CUDA graphs (seed 0), Pareto set marked."""
    pts = []  # (label, tokens, acc %, gpu ms, img/s at b64). Two fronts: batch-1 ms and batch-64 throughput
    for name, label, key in [("b0_full", "B0 224", "acc@0"), ("b0_448", "B0 native", "acc@0"), ("b0_native", "B0 native", "acc@0"),
                             ("b4_merge", "B4 merge", "acc@0"), ("b3_focusvit", "B3 FocusViT-style", "acc@1")]:
        accs = [r[key] for d in sorted(OUT.glob(f"{prefix}{name}/seed*")) if (r := _load(d / "summary_test.json")) and key in r]
        lat = _load(OUT / f"{prefix}{name}/seed0/latency_cuda_graph.json")
        if accs and lat:
            thr = _load(OUT / f"{prefix}{name}/seed0/latency_cuda_graph_b64.json")
            pts.append((label, "", 100 * np.mean(accs), lat["p50_ms"], thr and thr["img_per_s"]))
    for sweep, run, label in [("a3_keep_sweep", "b5_ours", "B5 ours"), ("a3_keep_sweep_par", "b5_par", "B5-par ours"),
                              ("keep_sweep", "b5_ours", "B5 ours"), ("keep_sweep_par", "b5_par", "B5-par ours")]:
        acc = defaultdict(list)
        for f in sorted(OUT.glob(f"{prefix}{sweep}/seed*/sweep.csv")):
            for r in csv.DictReader(open(f)):
                if r["rule"] == "fixed" and int(r["k_max"]) > 0:
                    acc[(float(r["keep_ratio"]), int(r["k_max"]))].append(float(r["acc"]))
        for (rho, k), a in sorted(acc.items()):
            lat = _load(OUT / f"{prefix}{run}/seed0/latency_cuda_graph_rho{rho}_k{k}.json")
            if lat:
                thr = _load(OUT / f"{prefix}{run}/seed0/latency_cuda_graph_b64_rho{rho}_k{k}.json")
                g = max(1, round((rho * 196) ** 0.5))
                pts.append((f"{label} ρ={rho} K={k}", f"{k}×{g * g}", 100 * np.mean(a), lat["p50_ms"], thr and thr["img_per_s"]))
    if not pts:
        return "_no trade-off runs yet_"
    front_ms = lambda p: not any(q[2] >= p[2] and q[3] <= p[3] and (q[2] > p[2] or q[3] < p[3]) for q in pts)  # noqa: E731
    thr = [q for q in pts if q[4]]
    front_thr = lambda p: p[4] is not None and not any(q[2] >= p[2] and q[4] >= p[4] and (q[2] > p[2] or q[4] > p[4]) for q in thr)  # noqa: E731
    fmt = lambda v, f: "–" if v is None else format(v, f)  # noqa: E731
    rows = [[("**" + p[0] + "**") if front_ms(p) or front_thr(p) else p[0], p[1], f"{p[2]:.2f}", f"{p[3]:.2f}", fmt(p[4], ".0f"),
             "✓" if front_ms(p) else "", "✓" if front_thr(p) else ""] for p in sorted(pts, key=lambda p: -p[2])]
    return _table(["Model / operating point", "Tokens", "Test acc %", "GPU ms (b1, graph)", "img/s (b64, graph)", "Pareto ms", "Pareto img/s"], rows)


def _compare() -> str:
    rows = []
    for p in sorted((OUT / "compare").glob("*.json")):
        r = json.loads(p.read_text())
        a = r["all"]
        strata = ", ".join(f"{k} {100 * r[k]['diff']:+.2f}" for k in ("coarse_wrong", "small", "large") if k in r)
        rows.append([p.stem, f"{r['rule_a']} → {r['rule_b']}", f"{100 * r['acc_a']:.2f} → {100 * r['acc_b']:.2f}",
                     f"{100 * a['diff']:+.2f} [{100 * a['lo']:+.2f}, {100 * a['hi']:+.2f}]", f"{a['p_le_0']:.3f}", strata])
    return _table(["Comparison (B − A)", "Rules", "Acc %", "Δ pt [95% CI]", "P(Δ ≤ 0)", "Δ by stratum"], rows) if rows else "_no comparisons yet_"


def write(path: str = "docs/CURRENT_RESULTS.md") -> None:
    runs = _runs()
    text = f"""# Current results

Generated by `make report` at {datetime.now():%Y-%m-%d %H:%M} from `output/`. Do not edit by hand.
Platform: see the Device column (phase 1 latency on RTX 3090 hitcvsz4, phase 2 on RTX 5090 D newpc, bf16 autocast).
Accuracy is on the test split; mean ± std over the seeds listed. acc@k = after k looks, with acc@0 = coarse pass (or the full model for B0/B4).

符号：ρ 是每次细看的 token 预算，相对整图 196 个 patch。每次 token 数是 g²，g = round(√(ρ·196))，所以 ρ=0.15 / 0.25 / 0.5 / 1 分别是 25 / 49 / 100 / 196。K 是最多细看几次。表里的 `K×N` 只数细看，粗看的 196 个 patch 另计，而且只跑前 2 层。研究问题用仓库里的编号：H1 往哪看，H2 何时停，H3 同一个权重在测试时改 (ρ, K, τ) 之后的准确率–时间前沿。

B0 到 B5 都是同一个分类任务上的不同做法：输入一张图，输出是哪一类。差别只在于计算花在整张图上，还是花在几块局部上。我们的模型是 B5。表里的前缀是换数据集，方法不变：没有前缀是 CUB 鸟，`canvas-` 是合成小目标，`cifar-` 是 CIFAR-100。`b1_random_on_b5` 表示选区改成随机，但网络权重仍用 B5 的，这样比的只是「往哪看」。

| 名字 | 它怎么看一张图 |
|---|---|
| **B0** | 整张图缩小后一次看完。`b0_full` 缩到 224²，`b0_448` 保持 448²。这是整图基线 |
| **B1** | 先粗看一遍，再随机指几块去细看 |
| **B2** | 同样细看几块，位置改成图像里最显眼的地方 |
| **B3** | 不裁局部。整张图打分，只把分数最高的大约 30% 小块送进 ViT，只看一次 |
| **B4** | 整张图都看。走到网络深层时，把相邻小块两两合成一个，后面算得少一些 |
| **B5** | 我们的模型。先粗看，再自己学该看哪一块、裁多大，看完一块再选下一块。默认看 3 次，每次 25 个 token |
| **B5-par** | 还是 B5。3 块在粗看之后一次选完，一起送进 ViT |

## Runs

这一节是实验目录，不单独回答一个研究问题。每一行是 `configs/exp/` 里的一个配置。q 是从粗分类得到的任务向量（`coarse_logits`；`none` 是消融 A5）。Role 里 `train` 表示这个配置自己训练，`eval of …` 表示只换推理时的选区策略、沿用另一个实验的权重。Seeds 的 ✓ 表示该种子已跑完。

参与的模型：`b0_full` / `b0_448` 是整图 ViT（224² 与原分辨率）；`b4_merge` 是整图上把相邻 token 合成更少的几个；`b3_focusvit` 是整图看一次、学习式 top-K；`b5_ours` 与 `b5_par` 是我们的顺序版和并行版；`b1_random`、`b2_saliency`、`oracle_gt` 以及带 `_on_b5` 的行是选区对照。`canvas-`、`canvas768-`、`cifar-` 是同一套模型换数据集。CUB、512² Canvas 和 768² Canvas 的主实验是三个种子，CIFAR-100 只有种子 0。

{_registry(runs)}

## Test accuracy (%)

指标是测试集 top-1，看满 K 次，不用 τ 提前停。`acc@k` 是细看 k 次之后的准确率。`acc@0` 对 B0、B4 就是整图模型的结果；对带注视的模型只是粗看（CUB 上大约 8%）。数字是种子均值 ± 标准差。

对应 H1（往哪看有没有用），以及各数据集上相对整图基线的准确率。CUB、默认每次 25 个 token：B0-224 76.30%，B0 原分辨率 84.08%，B4 70.96%，B3 的 `acc@1` 55.33%。`b5_ours` 看 1/2/3 次是 58.58% / 64.22% / 66.44%，`b5_par` 是 59.26% / 64.73% / 66.43%。同一套 B5 权重、看 3 次：随机 39.52%，显著性 50.86%，真值位置 68.83%。512² Canvas 上 `canvas-b5_ours` 只看 1 次是 39.62%，高于 `canvas-b0_full` 的 34.56%。CIFAR-100 上物体铺满画面，B0 86.15%，B5 看 3 次 76.30%。A1（关掉平移位置编码）、A5（关掉 q）、A6（改回 top-K）也在这张表里。

**一次细看在做什么。** 策略在粗看的 14×14 格子上选一格，再选边长（CUB 上是 80、112、160 或 224 像素）。按这个中心和边长，从原分辨率图上裁出一个正方形，缩成 g×g 个 patch（ρ=0.15 时是 5×5 = 25 个）。浅卷积给每个 patch 一个很小的平移，然后这 g² 个 token 走完 ViT 的全部 12 层，取出 CLS。GRU 把它和已有记忆合在一起，得到新的类别预测。`acc@k` 是前 k 块都已经写进 GRU 之后的 top-1，不是「只看第 k 块」。`b5_par` 把 K 个位置在粗看之后一次选完、一起过 ViT，GRU 仍按顺序累加，所以 `acc@k` 的含义不变。B0 和 B4 没有这个循环，只有 `acc@0`。

**第一列。** 没有前缀的是 CUB。`canvas-` 是 512² 合成画布，`canvas768-` 是 768²，`cifar-` 是 CIFAR-100。

| 实验 | 含义 |
|---|---|
| `b0_full` | 整图缩到 224²，196 个 token 走完 12 层 |
| `b0_448`、`canvas-b0_native`、`canvas768-b0_native` | 整图保持原分辨率 |
| `b4_merge` | 整图，深层把相邻 token 两两合成更少的几个 |
| `b5_ours` | 我们的模型：学出来的选区，看完一块再选下一块 |
| `b5_par` | 同一套训练，K 个位置一次选完再一起算 |
| `b1_random` / `b2_saliency` | 随机或按显著性选格子，权重是 Stage A |
| `b1_random_on_b5` / `b2_saliency_on_b5` | 同样的随机或显著性，权重和 `b5_ours` 相同。和主模型比时看这两行 |
| `oracle_gt` / `oracle_gt_on_b5` | 用标注框和身体部位选格子。带 `_on_b5` 的才是同权重上界 |
| `b3_focusvit` | 整张图只看一次，学习式 top-K 留下约 30% 的 patch |
| `a1_no_offset` | 和 B5 一样训练，去掉「patch 平移了多少」那一项位置编码 |
| `a2_fixed_k` | 就是 `b5_ours` 的权重，规定看满 3 次 |
| `a4_coarse_only` | 同一权重，K=0，一次细看都不做 |
| `a5_no_task` | 策略看不到粗分类 q，其余和 B5 一样重新训练 |
| `a6_topk_stage_a` | 细看不把整块缩成 5×5，改回给 patch 打分再取 top-K |

**Canvas、CIFAR、oracle。** Canvas 把一张 CIFAR-100 小图贴到大画布上，目标只有 32–56 像素，周围是打乱的图块和低频噪声，类别仍是 100 类。512² 与 768² 的目标像素数相同，所以在 768² 上更小；整图缩到 224² 之后目标会糊掉，细看原图上的那一块才有用。CIFAR 是把 CIFAR-100 直接放大到 224²，物体铺满画面，用来确认「没有局部可找时，细看不会更好」。Oracle 不是数据，也不是另一个网络：裁块、ViT 和 GRU 仍是我们的，只是第一次对准标注框、之后对准标注部位。Canvas 和 CIFAR 没有部位标注，所以没有 oracle 行。

{_accuracy(runs)}

## Operating points (τ* calibrated on val, reported on test)

参数是推理时的 ρ 和 τ，权重不重训。对每个 ρ 先看满 K 次，再在验证集上选最大的 τ*，使准确率仍不低于看满 K（容差 `eval.tol`）。下表是这个 τ* 在测试集上的结果。Rel. cost 是相对 B0-224 的 token-block 代价，1 表示和整图 12 层 × 196 个 token 相当。

模型是已经训好的 B5：`a3_keep_sweep` 用 `b5_ours`，`a3_keep_sweep_par` 用 `b5_par`。`canvas-keep_sweep` 与 `canvas768-keep_sweep` 是另外两个数据集上的同一次扫描。

对应 H2 和 H3。H2 问提前停能不能少算还不掉点：CUB 上 τ* 多在 0.1–0.4，平均注视仍大约 2–3 次。H3 问一个 checkpoint 换预算：ρ=0.15 时测试准确率约 66%、代价约 0.5×B0；ρ=0.5 时约 80%、代价约 1.2–1.7×B0。

{_sweep(runs)}

## Latency (batch 1)

指标是时间。每一行是某个模型在给定 ρ、K、τ 下的测速。`graph` 是 CUDA Graph，`eager` 是普通 PyTorch；`b64` 是 batch 64，没有 `b64` 的是 batch 1。p50 / p95 在 batch 1 的行上是单张延迟，在 `b64` 行上是整批的时间。img/s 是吞吐，GFLOPs 是每张图的浮点量。分模块毫秒只在 eager 下有。

对应 H3 的时间部分，以及 T1（CUDA Graph 是否去掉启动开销）。CUB、RTX 5090 D、ρ=0.15、K=3、τ=0：B0-224 是 0.72 ms、2.51 GFLOPs；`b5_ours` 是 2.22 ms、1.38 GFLOPs；`b5_par` 是 1.30 ms、同样 1.38 GFLOPs。batch 64 吞吐分别是 19,394、13,272、16,717 张/秒。

{_latency(runs)}

## Accuracy vs time: CUB (Pareto set in bold)

准确率和时间放在同一行。Tokens 列是「细看次数 × 每次 token 数」。`3×196` 是 ρ=1、K=3；`1×25` 才是 ρ=0.15、K=1。准确率是三个种子的测试 top-1 均值；毫秒和吞吐是种子 0、CUDA Graph、bf16。加粗的行至少在「单张毫秒」或「batch 64 吞吐」其中一个前沿上：没有别的点又更准又更快。

模型是 B0-224、B0 原分辨率（448²）、B4，以及 B5 顺序版和并行版的各个 (ρ, K)。

对应 H3，实验 T2（顺序）和 T3（并行）。B0-224 是 76.30%、19,394 张/秒；B0 原分辨率是 84.08%、5,762 张/秒。`B5-par ρ=0.5 K=2`（`2×100`）是 79.09%、13,682 张/秒，准确率高于 B0-224，吞吐高于原分辨率 B0。默认的 `ρ=0.15 K=3`（`3×25`）是 66.4%，两项都低于 B0-224。

{_pareto("")}

## Accuracy vs time: canvas 512² (targets 32–56 px)

参数和上一张表相同。数据是 512² 的合成画布，目标只有 32–56 像素。模型是 B0-224、B0 原分辨率、B4、B5 和 B5-par。指标仍是测试 top-1、batch-1 p50、batch 64 吞吐。

对应「小目标上能否超过把整图缩小」，实验 T4，也是 H3 在小目标上的前沿。`B5 ρ=0.15 K=1`（`1×25`）是 39.62%、20,748 张/秒，B0-224 是 34.56%、17,166 张/秒。`B5-par ρ=0.5 K=3`（`3×100`）是 59.18%、9,850 张/秒，原分辨率 B0 是 59.67%、3,980 张/秒。

{_pareto("canvas-")}

## Accuracy vs time: canvas 768² (same targets, smaller fraction of the image)

目标仍是 32–56 像素，画布改成 768²，目标占画面更小。这一组只训了 B0-224、B0 原分辨率和 B5 顺序版，没有 B4，也没有 B5-par。准确率是三个种子；速度是种子 0，在 RTX 4090 上测的，不能和上一张表的 5090 数字横比。

对应 T5：画布变大时，B0 原分辨率的 token 数按边长平方增长，B5 每次细看的 token 数不随画布变。`B5 ρ=0.15 K=1` 是 31.61%、11,671 张/秒，B0-224 是 24.91%、9,863 张/秒。`B5 ρ=0.5 K=3` 是 51.77%、6,103 张/秒；原分辨率 B0 是 57.44%、979 张/秒。

{_pareto("canvas768-")}

## Hypothesis tests (paired bootstrap over test images, seeds pooled)

指标是逐张配对的准确率差。三个种子的测试图合在一起做 bootstrap。Δ 是 B 减 A，单位是百分点；方括号是 95% 置信区间。P(Δ≤0) 小，表示 B 稳定地高于 A。最后一列把差值拆开：粗看就分错的图、鸟面积最小的三分之一、最大的三分之一。

前三行是 H1，都是看 2 次、每次 25 个 token。显著性比随机高 13.65 个点；学习式选区比显著性高 17.68 个点 [16.59, 18.83]。`h1_task_signal` 是打开 q 相对关掉 q，差 −0.37，区间包含 0。

`h2_stop` 是 H2：τ=0.2 相对看满 3 次，准确率差 −0.04。最后一行是 T6：`b5_par` 在 ρ=0.5、K=2（每次 100 个 token）相对 B0-224，整体 +2.78 [2.16, 3.42]，小鸟 +7.94，大鸟 −2.30。

{_compare()}
"""
    Path(path).write_text(text)
    print(f"wrote {path} ({len(runs)} experiments)")
