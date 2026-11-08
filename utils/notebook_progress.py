"""Live training dashboard for Jupyter.

Usage:
    from utils.notebook_progress import TrainingDashboard
    dash = TrainingDashboard(refresh_s=5)
    train_model(config=..., source_path=..., progress_callback=dash)

train.py calls dash(event, info) on every iteration plus on setup stages, evals,
saves, checkpoints, and the switch from the probabilistic phase to refinement.
The dashboard only syncs the GPU / copies images when it actually redraws
(every `refresh_s` seconds), so the overhead is small.
"""
import glob
import html
import os
import time
from collections import deque
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, Image, display

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _fmt_dur(s):
    if s is None or not np.isfinite(s):
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def _fmt_bytes(b):
    return f"{b / 2**30:.1f} GB"


def _cgroup_memory():
    """(current, limit) bytes of the nearest cgroup with a memory limit (the SLURM job), or None."""
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read().strip().split("\n")[0].split("::")[-1]
        parts = path.strip("/").split("/")
        for i in range(len(parts), 0, -1):
            d = os.path.join("/sys/fs/cgroup", *parts[:i])
            mx = os.path.join(d, "memory.max")
            if os.path.exists(mx):
                lim = open(mx).read().strip()
                if lim != "max":
                    cur = int(open(os.path.join(d, "memory.current")).read().strip())
                    return cur, int(lim)
    except Exception:
        pass
    return None


def _to_preview(img, max_w):
    img = img.detach()[:3].float().clamp(0, 1)
    if img.shape[-1] > max_w:
        h = int(img.shape[-2] * max_w / img.shape[-1])
        img = F.interpolate(img[None], size=(h, max_w), mode="area")[0]
    return img.permute(1, 2, 0).cpu().numpy()


class TrainingDashboard:
    def __init__(self, refresh_s=5.0, preview_width=480, show_test_grid=True, ram_warn=0.85):
        self.refresh_s = refresh_s
        self.preview_width = preview_width
        self.show_test_grid = show_test_grid
        self.ram_warn = ram_warn

        self.t_created = time.perf_counter()
        self.t_start = None            # wall clock at first training iteration
        self.extra_s = 0.0             # time spent in evals/saves/checkpoints/phase switch
        self.stage = "Waiting for train_model()"
        self.cfg = {}
        self.last_draw = 0.0
        self.error = None

        self.pending_losses = []       # (iter, loss tensor) not yet pulled to the CPU
        self.loss_hist = []            # (iter, loss)
        self.gauss_hist = []           # (iter, num_gaussians)
        self.stage_hist = []           # (iter, {stage: ms})
        self.evals = []                # (iter, {"test": {...}, "train": {...}})
        self.eval_secs, self.save_secs = [], []
        self.rate = {"probabilistic": deque(maxlen=300), "refinement": deque(maxlen=300)}
        self.events = deque(maxlen=12)
        self.last_iter_info = None
        self.phase_switch_iter = None
        self.busy_iter = None          # iteration that also ran an eval/save (its timings aren't representative)

        self.h_header = display(HTML("<i>Dashboard ready - waiting for training to start...</i>"), display_id=True)
        self.h_fig = display(HTML(""), display_id=True)
        self.h_grid = display(HTML(""), display_id=True)
        self.h_log = display(HTML(""), display_id=True)

        self.fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
        self.ax_img, self.ax_gt, self.ax_err = axs[0]
        self.ax_loss, self.ax_psnr, self.ax_time = axs[1]
        self.ax_gauss = self.ax_time.twinx()
        plt.close(self.fig)            # don't let Jupyter auto-display it

    # ------------------------------------------------------------------ events
    def __call__(self, event, info):
        try:
            self._handle(event, info)
        except Exception as e:  # never let the dashboard kill a training run
            if self.error is None:
                self.error = repr(e)
                self._log(f"<b style='color:#c00'>dashboard error (training continues): {html.escape(self.error)}</b>")
                try:
                    self.h_log.update(HTML(self._log_html()))
                except Exception:
                    pass

    def _log(self, msg):
        self.events.appendleft(f"<code>{datetime.now():%H:%M:%S}</code> {msg}")

    def _handle(self, event, info):
        now = time.perf_counter()
        if event == "stage":
            self.stage = info["msg"]
            self._log(html.escape(info["msg"]))
            return self._redraw(force=True, figure=False)

        if event == "start":
            self.cfg = info
            self.t_start = now
            self.stage = "Training"
            self._log(f"Training started: {info['n_train']} train / {info['n_test']} test views, "
                      f"{info['prob_iters']:,} probabilistic + {info['refinement_iters']:,} refinement iters, "
                      f"{len(info['eval_iters'])} evals scheduled. Output: <code>{html.escape(info['model_path'])}</code>")
            return self._redraw(force=True, figure=False)

        if event == "iter":
            it = info["iteration"]
            self.last_iter_info = info
            self.pending_losses.append((it, info["loss"].detach()))
            self.rate[info["phase"]].append((it, now - self.extra_s))
            if now - self.last_draw >= self.refresh_s or it == info["last_iter"]:
                self.gauss_hist.append((it, info["num_gaussians"]))
                if it != self.busy_iter:   # skip iterations whose "step" stage includes an eval/save
                    self.stage_hist.append((it, info["timings"]()))
                self._redraw(force=True)
            return

        if event == "eval":
            self.extra_s += info["seconds"]
            self.busy_iter = info["iteration"]
            self.eval_secs.append(info["seconds"])
            self.evals.append((info["iteration"], info["results"]))
            r = info["results"]
            txt = " | ".join(f"{k}: PSNR {v['psnr']:.2f} SSIM {v['ssim']:.3f} LPIPS {v['lpips']:.3f}" for k, v in r.items())
            self._log(f"<b>Eval @ {info['iteration']:,}</b> ({_fmt_dur(info['seconds'])}): {txt}")
            if self.show_test_grid:
                grids = glob.glob(os.path.join(info["model_path"], f"test_visualization_{info['iteration']}_for_PSNR=*.png"))
                if grids:
                    self.h_grid.update(Image(filename=grids[0], width=1100))
            return self._redraw(force=True)

        if event in ("save", "checkpoint"):
            self.extra_s += info["seconds"]
            self.busy_iter = info["iteration"]
            self.save_secs.append(info["seconds"])
            what = "PLY + grids" if event == "save" else "checkpoint"
            self._log(f"Saved {what} @ {info['iteration']:,} ({_fmt_dur(info['seconds'])})")
            return self._redraw(force=True, figure=False)

        if event == "phase":
            self.phase_switch_iter = info["iteration"]
            self._log(f"<b>Switched to refinement @ {info['iteration']:,}</b>: froze "
                      f"{info['num_gaussians']:,} explicit Gaussians (plain 3DGS optimization from here)")
            # conversion time gets excluded from the per-iteration rate
            self.rate["refinement"].clear()
            return self._redraw(force=True, figure=False)

        if event == "end":
            self.stage = "Done"
            self._log(f"<b>Training finished</b> in {_fmt_dur(now - self.t_start)}. Model: <code>{html.escape(info['model_path'])}</code>")
            return self._redraw(force=True)

    # ------------------------------------------------------------- estimates
    def _sec_per_it(self, phase):
        d = self.rate[phase]
        if len(d) < 5:
            return None
        (i0, t0), (i1, t1) = d[0], d[-1]
        return (t1 - t0) / max(i1 - i0, 1)

    def _eta(self):
        if not self.cfg or self.last_iter_info is None:
            return None, None, ""
        it = self.last_iter_info["iteration"]
        P, L = self.cfg["prob_iters"], self.cfg["last_iter"]
        spi_p, spi_r = self._sec_per_it("probabilistic"), self._sec_per_it("refinement")
        notes = []
        rem_p, rem_r = max(P - it, 0), min(L - it, L - P)
        if spi_r is None and rem_r > 0:
            spi_r = spi_p  # refinement is usually faster; this is an upper bound
            notes.append("refinement speed not measured yet (assuming probabilistic speed)")
        if (rem_p and spi_p is None) or (rem_r and spi_r is None):
            return None, None, "measuring speed..."
        train_s = rem_p * (spi_p or 0) + rem_r * (spi_r or 0)
        rem_evals = sum(1 for e in self.cfg["eval_iters"] if e > it)
        rem_saves = sum(1 for s in self.cfg["save_iters"] if s > it)
        extra = 0.0
        if rem_evals:
            if self.eval_secs:
                extra += rem_evals * float(np.mean(self.eval_secs))
            else:
                notes.append(f"+ {rem_evals} evals (duration unknown until the first one)")
        if rem_saves and self.save_secs:
            extra += rem_saves * float(np.mean(self.save_secs))
        return train_s, train_s + extra, "; ".join(notes)

    # --------------------------------------------------------------- drawing
    def _redraw(self, force=False, figure=True):
        self.last_draw = time.perf_counter()
        if self.pending_losses:
            its = [i for i, _ in self.pending_losses]
            vals = torch.stack([l for _, l in self.pending_losses]).float().cpu().tolist()
            self.loss_hist.extend(zip(its, vals))
            self.pending_losses.clear()
        self._draw_header()
        if figure and self.last_iter_info is not None:
            self._draw_figure()
        self.h_log.update(HTML(self._log_html()))

    def _log_html(self):
        return ("<div style='font-family:monospace;font-size:12px;line-height:1.5'>"
                "<b>Event log</b> (newest first)<br>" + "<br>".join(self.events) + "</div>")

    def _draw_header(self):
        now = time.perf_counter()
        rows = []
        if self.last_iter_info is None:
            rows.append(f"<b>Stage:</b> {html.escape(self.stage)} &nbsp; "
                        f"<span style='color:#666'>(elapsed {_fmt_dur(now - self.t_created)}; first run also JIT-compiles CUDA/Slang kernels)</span>")
            pct, bar_label = 0.0, "setup"
        else:
            info = self.last_iter_info
            it, L = info["iteration"], info["last_iter"]
            P = self.cfg.get("prob_iters", L)
            pct = 100.0 * it / L
            phase = info["phase"]
            color = "#2a7ae2" if phase == "probabilistic" else "#e2832a"
            spi = self._sec_per_it(phase)
            train_eta, total_eta, note = self._eta()
            elapsed = now - self.t_start
            finish = (datetime.now() + timedelta(seconds=total_eta)).strftime("%a %H:%M") if total_eta is not None else "?"
            loss_ema = None
            if self.loss_hist:
                v = np.array([l for _, l in self.loss_hist[-200:]])
                loss_ema = float(v.mean())
            phase_txt = (f"{it:,} / {P:,} probabilistic" if phase == "probabilistic"
                         else f"{it - P:,} / {L - P:,} refinement")
            bar_label = f"{it:,} / {L:,} ({pct:.1f}%)"
            rows.append(f"<span style='background:{color};color:white;padding:2px 8px;border-radius:4px'>{phase.upper()}</span>"
                        f" &nbsp; <b>{phase_txt}</b> &nbsp; | &nbsp; {html.escape(self.stage)}")
            rows.append(f"<b>Elapsed</b> {_fmt_dur(elapsed)} &nbsp; <b>Speed</b> "
                        f"{(1 / spi if spi else float('nan')):.2f} it/s ({(spi or float('nan')) * 1000:.0f} ms/it) &nbsp; "
                        f"<b>ETA training</b> {_fmt_dur(train_eta)} &nbsp; <b>ETA incl. evals/saves</b> {_fmt_dur(total_eta)} "
                        f"(done ≈ {finish})" + (f" <span style='color:#888'>[{html.escape(note)}]</span>" if note else ""))
            if self.stage_hist:
                st = self.stage_hist[-1][1]
                tot = sum(st.values())
                rows.append("<b>Last iter breakdown</b> " + " · ".join(
                    f"{k} {v:.1f} ms ({100 * v / max(tot, 1e-9):.0f}%)" for k, v in st.items()))
            gs = f"{info['num_gaussians']:,}"
            rows.append(f"<b>Loss (avg last 200)</b> {loss_ema:.4f} &nbsp; <b>Gaussians this iter</b> {gs}"
                        if loss_ema is not None else f"<b>Gaussians this iter</b> {gs}")
            if self.evals:
                ei, er = self.evals[-1]
                rows.append(f"<b>Last eval @ {ei:,}</b> " + " | ".join(
                    f"{k}: PSNR {v['psnr']:.2f} SSIM {v['ssim']:.3f} LPIPS {v['lpips']:.3f}" for k, v in er.items()))
        rows.append(self._mem_html())
        bar = (f"<div style='background:#ddd;border-radius:4px;height:18px;width:100%;position:relative'>"
               f"<div style='background:linear-gradient(90deg,#2a7ae2,#e2832a);height:18px;border-radius:4px;width:{pct:.2f}%'></div>"
               f"<div style='position:absolute;top:0;width:100%;text-align:center;font-size:12px;line-height:18px'>{bar_label}</div></div>")
        self.h_header.update(HTML("<div style='font-size:13px;line-height:1.7'>" + bar + "<br>".join(rows) + "</div>"))

    def _mem_html(self):
        parts = []
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"<b>GPU</b> {_fmt_bytes(total - free)} / {_fmt_bytes(total)} used "
                         f"(torch peak {_fmt_bytes(torch.cuda.max_memory_allocated())})")
        cg = _cgroup_memory()
        if cg:
            cur, lim = cg
            frac = cur / lim
            style = "color:#c00;font-weight:bold" if frac > self.ram_warn else ""
            warn = " ⚠ close to the SLURM memory limit - job may be OOM-killed" if frac > self.ram_warn else ""
            parts.append(f"<span style='{style}'><b>Host RAM (job)</b> {_fmt_bytes(cur)} / {_fmt_bytes(lim)} ({100 * frac:.0f}%){warn}</span>")
        elif psutil is not None:
            vm = psutil.virtual_memory()
            parts.append(f"<b>Host RAM</b> {_fmt_bytes(vm.used)} / {_fmt_bytes(vm.total)}")
        return " &nbsp; ".join(parts)

    def _draw_figure(self):
        info = self.last_iter_info
        P = self.cfg.get("prob_iters")
        for ax in (self.ax_img, self.ax_gt, self.ax_err, self.ax_loss, self.ax_psnr, self.ax_time, self.ax_gauss):
            ax.clear()

        img = _to_preview(info["image"], self.preview_width)
        gt = _to_preview(info["gt"], self.preview_width)
        err = np.abs(img - gt).mean(-1)
        self.ax_img.imshow(img); self.ax_img.set_title(f"Render ({info['phase']}) - iter {info['iteration']:,}")
        self.ax_gt.imshow(gt); self.ax_gt.set_title(f"Ground truth - {info['cam_name']}")
        im = self.ax_err.imshow(err, cmap="inferno", vmin=0, vmax=0.3)
        self.ax_err.set_title(f"|error| (mean {err.mean():.3f})")
        for ax in (self.ax_img, self.ax_gt, self.ax_err):
            ax.axis("off")

        if self.loss_hist:
            it, l = map(np.array, zip(*self.loss_hist))
            self.ax_loss.plot(it, l, color="#bbb", lw=0.5, label="per-iter")
            k = max(1, min(200, len(l) // 20))
            if len(l) > k:
                sm = np.convolve(l, np.ones(k) / k, mode="valid")
                self.ax_loss.plot(it[k - 1:], sm, color="#2a7ae2", lw=1.5, label=f"mean of {k}")
            self.ax_loss.set_yscale("log")
        if P:
            for ax in (self.ax_loss, self.ax_psnr, self.ax_time):
                ax.axvline(P, color="#e2832a", ls="--", lw=1)
        self.ax_loss.set_title("Training loss (L1 + D-SSIM + regs)"); self.ax_loss.set_xlabel("iteration")
        self.ax_loss.legend(loc="upper right", fontsize=8)

        if self.evals:
            for split, c in (("test", "#d62728"), ("train", "#2ca02c")):
                pts = [(i, r[split]["psnr"]) for i, r in self.evals if split in r]
                if pts:
                    x, y = zip(*pts)
                    self.ax_psnr.plot(x, y, "o-", color=c, label=split)
                    self.ax_psnr.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
            self.ax_psnr.legend(fontsize=8)
        else:
            self.ax_psnr.text(0.5, 0.5, "no eval yet\n(next at iter %s)" % next(
                (e for e in self.cfg.get("eval_iters", []) if e > info["iteration"]), "?"),
                ha="center", va="center", transform=self.ax_psnr.transAxes, color="#888")
        self.ax_psnr.set_title("PSNR at evals"); self.ax_psnr.set_xlabel("iteration")

        if self.stage_hist:
            it = np.array([i for i, _ in self.stage_hist])
            names = list(self.stage_hist[0][1].keys())
            ys = np.array([[d[n] for n in names] for _, d in self.stage_hist]).T
            self.ax_time.stackplot(it, ys, labels=names, alpha=0.8)
            self.ax_time.set_ylabel("ms / iter (sampled)")
            self.ax_time.legend(loc="upper left", fontsize=7)
        if self.gauss_hist:
            gi, gn = zip(*self.gauss_hist)
            gn = np.array(gn) / 1e6
            self.ax_gauss.plot(gi, gn, "k-", lw=1, label="Gaussians")
            self.ax_gauss.set_ylim(0, gn.max() * 1.15)
            self.ax_gauss.ticklabel_format(axis="y", useOffset=False)
            self.ax_gauss.set_ylabel("Gaussians (M)")
            self.ax_gauss.yaxis.set_label_position("right")   # clear() on a twin axis resets this
            self.ax_gauss.yaxis.tick_right()
        self.ax_time.set_title("Per-stage GPU time + #Gaussians"); self.ax_time.set_xlabel("iteration")

        self.fig.tight_layout()
        self.h_fig.update(self.fig)
