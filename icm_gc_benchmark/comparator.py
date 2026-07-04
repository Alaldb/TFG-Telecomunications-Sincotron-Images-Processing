import os, sys, time, tracemalloc, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import permutations as perms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from processing.isingMethodService import Ising
from processing.graphCutsService import GraphCutsService


class BenchmarkComparator:

    K            = 3
    N_PER_GROUP  = 20
    CALIB_N      = 4
    IMG_H        = 256
    IMG_W        = 256
    MAX_ITER_ICM = 100
    BETA_GRID    = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    NGAUSS_GRID  = [1, 2, 3, 5]
    A3_GROUP     = "G3_SNR6"   # grupo fijo para approach 3
    A3_NGAUSS_GC = 3           # n_gaussians fijo para GC en approach 3

    GROUPS = [
        dict(A=0.80, sigma=0.02, name="G1_SNR40"),
        dict(A=0.55, sigma=0.04, name="G2_SNR14"),
        dict(A=0.35, sigma=0.06, name="G3_SNR6"),
        dict(A=0.20, sigma=0.08, name="G4_SNR2"),
        dict(A=0.10, sigma=0.10, name="G5_SNR1"),
    ]

    @staticmethod
    def _snr(gdef):
        return round(gdef["A"] / gdef["sigma"], 1)

    def __init__(self, real_folder: str, out_dir: str):
        self.real_folder  = real_folder
        self.out_dir      = out_dir
        self.database     = {}
        self.real_images  = []
        self.real_names   = []
        self.results_a1   = {}
        self.results_a2   = {}
        self.results_a3_icm = {}   # {beta: [metric_dicts]}
        self.results_a3_gc  = {}   # {beta: [metric_dicts]}
        for sub in ["", "approach1", "approach2", "approach3", "figures"]:
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # GENERADOR IMÁGENES
    # ══════════════════════════════════════════════════════════════════════════
    def generate_images(self, seed=None, sigma_filter=40.0,
                         area_pct_pos=8.0, area_pct_neg=8.0,
                         A_signal=0.8, sigma_noise=0.06):
        H, W     = self.IMG_H, self.IMG_W
        rng      = np.random.default_rng(seed)
        noise    = rng.normal(0.0, 1.0, (H, W))
        F        = np.fft.fft2(noise)
        fy       = np.fft.fftfreq(H)[:, None]
        fx       = np.fft.fftfreq(W)[None, :]
        r        = np.sqrt(fx**2 + fy**2)
        lp       = np.exp(-(r**2) / (2 * (sigma_filter / max(H, W))**2))
        filtered = np.real(np.fft.ifft2(F * lp))
        thr_pos  = np.percentile(filtered, 100.0 - area_pct_pos)
        thr_neg  = np.percentile(filtered, area_pct_neg)
        mask     = np.zeros((H, W), dtype=np.float32)
        mask[filtered > thr_pos] = +1.0
        mask[filtered < thr_neg] = -1.0
        mask_s   = cv2.GaussianBlur(mask, (5, 5), 1.0)
        image    = mask_s * A_signal + rng.normal(0.0, sigma_noise, (H, W))
        return image.astype(np.float32), mask.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # MÉTRICAS
    # ══════════════════════════════════════════════════════════════════════════
    def mask_to_labels(self, gt_mask):
        gt = np.zeros_like(gt_mask, dtype=np.int32)
        gt[gt_mask > 0.5]  = 2
        gt[gt_mask < -0.5] = 1
        return gt

    def align_labels(self, pred, gt):
        K     = self.K
        valid = (gt.ravel() >= 0) & (pred.ravel() >= 0)
        best_acc, best_perm = -1.0, list(range(K))
        for perm in perms(range(K)):
            actual_perm  = np.array(perm)
            acc = float((actual_perm[pred.ravel()[valid]] == gt.ravel()[valid]).mean())
            if acc > best_acc:
                best_acc, best_perm = acc, list(perm)
        return np.array(best_perm)[np.clip(pred, 0, K-1)].astype(np.int32)

    def compute_metrics(self, pred_raw, gt_mask, image):
        K     = self.K
        gt    = self.mask_to_labels(gt_mask)
        pred  = self.align_labels(pred_raw, gt)
        valid = (gt >= 0) & (pred >= 0)

        acc  = float((pred[valid] == gt[valid]).mean())

        ious = []
        for k in range(K):
            inter = ((pred == k) & (gt == k) & valid).sum()
            union = (((pred == k) | (gt == k)) & valid).sum()
            if union > 0:
                ious.append(inter / union)
        miou = float(np.mean(ious)) if ious else 0.0

        dice = []
        for k in range(K):
            tp = ((pred == k) & (gt == k) & valid).sum()
            fp = ((pred == k) & (gt != k) & valid).sum()
            fn = ((pred != k) & (gt == k) & valid).sum()
            d  = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 1.0
            dice.append(float(d))

        residuals = []
        for k in range(K):
            m = (gt == k) & valid
            if m.sum() > 1:
                residuals.append(float(image[m].std()))
        noise_res = float(np.mean(residuals)) if residuals else 0.0

        return {"accuracy": acc, "mean_iou": miou,
                "dice_mean": float(np.mean(dice)),
                "noise_residual": noise_res}

    def compute_blind_metrics(self, labels, image):
        K     = self.K
        valid = labels >= 0
        vars = []
        for k in range(K):
            m = (labels == k) & valid
            if m.sum() > 1:
                vars.append(float(image[m].var()))
        intra_var = float(np.mean(vars)) if vars else 0.0
        counts    = np.array([(labels == k).sum() for k in range(K)], dtype=float)
        counts   /= counts.sum() + 1e-12
        entropy   = float(-np.sum(counts * np.log(counts + 1e-12)))
        return {"intra_var": intra_var, "entropy": entropy}

    # ══════════════════════════════════════════════════════════════════════════
    # EJECUTAR CON TIMING Y MEMORIA
    # ══════════════════════════════════════════════════════════════════════════
    def run_icm(self, image, beta):
        solver = Ising(beta=beta, max_iterations=self.MAX_ITER_ICM,
                       num_states=self.K)
        tracemalloc.start()
        t0 = time.perf_counter()
        solver.run(image)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return solver.final_image, {"time_s": elapsed, "memory_mb": peak / 1e6}

    def _run_gc(self, image, lambda_value, n_gaussians):
        solver = GraphCutsService(num_states=self.K,
                                  lambda_value=lambda_value,
                                  sigma=None,
                                  num_iterations=-1,
                                  number_gaussians_per_state=n_gaussians)
        tracemalloc.start()
        t0 = time.perf_counter()
        solver.run(image)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return solver.final_image, {"time_s": elapsed, "memory_mb": peak / 1e6}

    # ══════════════════════════════════════════════════════════════════════════
    # CALIBRACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    def iou_icm(self, imgs, masks, beta):
        scores = []
        for img, mask in zip(imgs, masks):
            lbl, _ = self.run_icm(img, beta)
            scores.append(self.compute_metrics(lbl, mask, img)["mean_iou"])
        return float(np.mean(scores))

    def iou_gc(self, imgs, masks, lam, n_gauss):
        scores = []
        for img, mask in zip(imgs, masks):
            lbl, _ = self._run_gc(img, lam, n_gauss)
            scores.append(self.compute_metrics(lbl, mask, img)["mean_iou"])
        return float(np.mean(scores))

    def calibrate_shared(self, imgs, masks):
        best_b, best_s = self.BETA_GRID[0], -1.0
        for b in self.BETA_GRID:
            s = (self.iou_icm(imgs, masks, b) +
                 self.iou_gc(imgs, masks, b, 1)) / 2
            print(f"beta={b:.1f} -> score={s:.4f}")
            if s > best_s:
                best_b, best_s = b, s
        return best_b

    def calibrate_icm(self, imgs, masks):
        best_b, best_s = self.BETA_GRID[0], -1.0
        for b in self.BETA_GRID:
            s = self.iou_icm(imgs, masks, b)
            if s > best_s:
                best_b, best_s = b, s
        return best_b

    def calibrate_gc(self, imgs, masks):
        best_b, best_ng, best_s = self.BETA_GRID[0], 1, -1.0
        for b in self.BETA_GRID:
            for ng in self.NGAUSS_GRID:
                s = self.iou_gc(imgs, masks, b, ng)
                if s > best_s:
                    best_b, best_ng, best_s = b, ng, s
        return best_b, best_ng

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZACIÓN — helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _disp(self, image):
        lo, hi = np.percentile(image, 1), np.percentile(image, 99)
        return np.clip((image - lo) / (hi - lo + 1e-9), 0, 1)

    def _save_synth_fig(self, image, gt_mask, lbl_icm, lbl_gc, path, title=""):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(self._disp(image), cmap='gray');                axes[0].set_title("Original")
        axes[1].imshow(gt_mask, cmap='RdBu_r', vmin=-1, vmax=1);      axes[1].set_title("Ground truth")
        axes[2].imshow(lbl_icm, cmap='tab10', vmin=0, vmax=self.K-1); axes[2].set_title("ICM")
        axes[3].imshow(lbl_gc,  cmap='tab10', vmin=0, vmax=self.K-1); axes[3].set_title("GC")
        for ax in axes: ax.axis('off')
        if title: fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_real_fig(self, image, lbl_icm, lbl_gc, path, title=""):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(self._disp(image), cmap='gray');                axes[0].set_title("Original")
        axes[1].imshow(lbl_icm, cmap='tab10', vmin=0, vmax=self.K-1); axes[1].set_title("ICM")
        axes[2].imshow(lbl_gc,  cmap='tab10', vmin=0, vmax=self.K-1); axes[2].set_title("GC")
        for ax in axes: ax.axis('off')
        if title: fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_histograms(self, results_by_group, metric, label, out_dir):
        snr_vals = [self._snr(g) for g in self.GROUPS]
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for g, (gname, gdata) in enumerate(results_by_group.items()):
            icm_vals = [r[metric] for r in gdata["icm"] if metric in r]
            gc_vals  = [r[metric] for r in gdata["gc"]  if metric in r]
            axes[g].hist(icm_vals, bins=8, alpha=0.6, label='ICM', color='steelblue')
            axes[g].hist(gc_vals,  bins=8, alpha=0.6, label='GC',  color='coral')
            axes[g].set_title(f"SNR={snr_vals[g]}", fontsize=9)
            axes[g].set_xlabel(metric, fontsize=7)
            axes[g].legend(fontsize=7)
        fig.suptitle(f"{label} — {metric}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{label}_{metric}.png"),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_summary(self, results_by_group, label, out_dir):
        metrics = ["accuracy", "mean_iou", "dice_mean",
                   "noise_residual", "time_s", "memory_mb"]
        rows = []
        for gname, gdata in results_by_group.items():
            for method, key in [("ICM", "icm"), ("GC", "gc")]:
                row = {"group": gname, "method": method}
                for m in metrics:
                    vals = [r[m] for r in gdata[key] if m in r]
                    row[f"{m}_mean"] = round(float(np.mean(vals)), 5) if vals else float('nan')
                    row[f"{m}_std"]  = round(float(np.std(vals)),  5) if vals else float('nan')
                rows.append(row)
        if rows:
            with open(os.path.join(out_dir, f"summary_{label}.csv"), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    # Approaches 1 y 2 — una figura por metrica, eje X = SNR
    def _save_approach_figures(self, results, approach_label, out_dir):
        group_names = [g["name"] for g in self.GROUPS]
        snr_vals    = [self._snr(g) for g in self.GROUPS]
        metrics     = ["accuracy", "mean_iou", "dice_mean", "noise_residual", "time_s", "memory_mb"]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method, color, key in [("ICM", "steelblue", "icm"),
                                        ("GC",  "coral",     "gc")]:
                means = [np.mean([r[metric] for r in results[g][key] if metric in r])
                         for g in group_names]
                stds  = [np.std( [r[metric] for r in results[g][key] if metric in r])
                         for g in group_names]
                ax.errorbar(snr_vals, means, yerr=stds, label=method,
                            marker='o', color=color, capsize=4)
            ax.set_xlabel("SNR (A / sigma)", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{approach_label} — {metric}", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{approach_label}_{metric}.png"),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

    # Approach 3 — una figura por metrica, eje X = beta
    def _save_approach3_figures(self, out_dir):
        metrics = ["accuracy", "mean_iou", "dice_mean", "noise_residual", "time_s", "memory_mb"]
        betas   = self.BETA_GRID

        for metric in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, (method_label, results) in zip(
                    axes, [("ICM", self.results_a3_icm),
                           ("GC",  self.results_a3_gc)]):
                means = [np.mean([r[metric] for r in results[b] if metric in r])
                         for b in betas]
                stds  = [np.std( [r[metric] for r in results[b] if metric in r])
                         for b in betas]
                color = "steelblue" if method_label == "ICM" else "coral"
                ax.errorbar(betas, means, yerr=stds, marker='o',
                            color=color, capsize=4, label=method_label)
                ax.set_xlabel("beta / lambda", fontsize=10)
                ax.set_ylabel(metric, fontsize=10)
                ax.set_title(f"Approach 3 {method_label} — {metric}\n"
                             f"(grupo fijo: {self.A3_GROUP})", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"approach3_{metric}.png"),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

    def _save_a3_summary(self, out_dir):
        metrics = ["accuracy", "mean_iou", "dice_mean", "noise_residual", "time_s", "memory_mb"]
        rows = []
        for method_label, results in [("ICM", self.results_a3_icm),
                                       ("GC",  self.results_a3_gc)]:
            for beta, rlist in results.items():
                row = {"method": method_label, "beta": beta}
                for m in metrics:
                    vals = [r[m] for r in rlist if m in r]
                    row[f"{m}_mean"] = round(float(np.mean(vals)), 5) if vals else float('nan')
                    row[f"{m}_std"]  = round(float(np.std(vals)),  5) if vals else float('nan')
                rows.append(row)
        if rows:
            with open(os.path.join(out_dir, "summary_approach3.csv"), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE PÚBLICO
    # ══════════════════════════════════════════════════════════════════════════
    def generate_database(self):
        print("[1/6] Generando 100 imagenes sinteticas...")
        seed_base = 0
        for g, gdef in enumerate(self.GROUPS):
            imgs, masks = [], []
            for i in range(self.N_PER_GROUP):
                img, mask = self.generate_images(
                    seed=seed_base + i,
                    A_signal=gdef["A"],
                    sigma_noise=gdef["sigma"])
                imgs.append(img)
                masks.append(mask)
            self.database[gdef["name"]] = {"images": imgs, "masks": masks, **gdef}
            seed_base += self.N_PER_GROUP
            snr = self._snr(gdef)
            print(f"   G{g+1} {gdef['name']}: A={gdef['A']}, sigma={gdef['sigma']}, SNR={snr}")

    def load_real_images(self):
        print("[2/6] Cargando imagenes reales...")
        from PIL import Image as PILImage
        exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        for fname in sorted(os.listdir(self.real_folder)):
            if os.path.splitext(fname)[1].lower() in exts:
                arr = np.array(PILImage.open(
                    os.path.join(self.real_folder, fname)), dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=2)
                self.real_images.append(arr)
                self.real_names.append(fname)
        print(f"   {len(self.real_images)} imagenes cargadas")

    def run_approach1(self):
        print("[3/6] Approach 1 - condiciones iguales (n_gaussians=1)...")
        g1   = self.database[self.GROUPS[0]["name"]]
        print("   Calibrando beta compartida...")
        beta = self.calibrate_shared(g1["images"][:self.CALIB_N],
                                      g1["masks"][:self.CALIB_N])
        print(f"   Beta compartida = {beta}")
        out = os.path.join(self.out_dir, "approach1")

        for gname, gdata in self.database.items():
            print(f"   {gname}...", end=" ", flush=True)
            icm_r, gc_r = [], []
            for idx, (img, mask) in enumerate(
                    zip(gdata["images"][self.CALIB_N:],
                        gdata["masks"][self.CALIB_N:])):
                li, ri = self.run_icm(img, beta)
                lg, rg = self._run_gc(img, beta, n_gaussians=1)
                icm_r.append({**self.compute_metrics(li, mask, img), **ri})
                gc_r.append( {**self.compute_metrics(lg, mask, img), **rg})
                if idx == 0:
                    self._save_synth_fig(img, mask, li, lg,
                        os.path.join(out, f"{gname}_img0.png"),
                        title=f"Approach 1 | {gname} | beta={beta}")
            self.results_a1[gname] = {"icm": icm_r, "gc": gc_r}
            print("ok")

        self._save_summary(self.results_a1, "approach1", out)
        self._save_approach_figures(self.results_a1, "approach1",
                                    os.path.join(self.out_dir, "figures"))
        for m in ["accuracy", "mean_iou", "time_s"]:
            self._save_histograms(self.results_a1, m, "approach1", out)

        print("   Imagenes reales (approach 1)...")
        for img, fname in zip(self.real_images, self.real_names):
            li, _ = self.run_icm(img, beta)
            lg, _ = self._run_gc(img, beta, n_gaussians=1)
            self._save_real_fig(img, li, lg,
                os.path.join(out, f"real_{fname}.png"),
                title=f"Approach 1 | {fname} | beta={beta}")

    def run_approach2(self):
        print("[4/6] Approach 2 - best effort...")
        out = os.path.join(self.out_dir, "approach2")

        for gname, gdata in self.database.items():
            calib_i = gdata["images"][:self.CALIB_N]
            calib_m = gdata["masks"][:self.CALIB_N]
            bi      = self.calibrate_icm(calib_i, calib_m)
            bg, ng  = self.calibrate_gc(calib_i, calib_m)
            print(f"   {gname}: beta_ICM={bi}, beta_GC={bg}, n_gauss={ng}")

            icm_r, gc_r = [], []
            for idx, (img, mask) in enumerate(
                    zip(gdata["images"][self.CALIB_N:],
                        gdata["masks"][self.CALIB_N:])):
                li, ri = self.run_icm(img, bi)
                lg, rg = self._run_gc(img, bg, n_gaussians=ng)
                icm_r.append({**self.compute_metrics(li, mask, img), **ri})
                gc_r.append( {**self.compute_metrics(lg, mask, img), **rg})
                if idx == 0:
                    self._save_synth_fig(img, mask, li, lg,
                        os.path.join(out, f"{gname}_img0.png"),
                        title=f"Approach 2 | {gname} | beta_ICM={bi} beta_GC={bg} ng={ng}")
            self.results_a2[gname] = {"icm": icm_r, "gc": gc_r,
                                      "beta_icm": bi, "beta_gc": bg, "n_gauss": ng}

        self._save_summary(self.results_a2, "approach2", out)
        self._save_approach_figures(self.results_a2, "approach2",
                                    os.path.join(self.out_dir, "figures"))
        for m in ["accuracy", "mean_iou", "time_s"]:
            self._save_histograms(self.results_a2, m, "approach2", out)

        print("   Imagenes reales (approach 2)...")
        g1         = self.database[self.GROUPS[0]["name"]]
        bi_r       = self.calibrate_icm(g1["images"][:self.CALIB_N], g1["masks"][:self.CALIB_N])
        bg_r, ng_r = self.calibrate_gc( g1["images"][:self.CALIB_N], g1["masks"][:self.CALIB_N])
        for img, fname in zip(self.real_images, self.real_names):
            li, _ = self.run_icm(img, bi_r)
            lg, _ = self._run_gc(img, bg_r, n_gaussians=ng_r)
            self._save_real_fig(img, li, lg,
                os.path.join(out, f"real_{fname}.png"),
                title=f"Approach 2 | {fname}")

    def run_approach3(self):
        print("[5/6] Approach 3 - barrido beta con nivel de ruido fijo "
              f"({self.A3_GROUP})...")
        out     = os.path.join(self.out_dir, "approach3")
        gdata   = self.database[self.A3_GROUP]
        imgs    = gdata["images"][self.CALIB_N:]
        masks   = gdata["masks"][self.CALIB_N:]

        # Barrido ICM
        print("   ICM sweep...")
        for beta in self.BETA_GRID:
            print(f"     beta={beta:.1f}", end=" ", flush=True)
            rlist = []
            for idx, (img, mask) in enumerate(zip(imgs, masks)):
                lbl, rinfo = self.run_icm(img, beta)
                rlist.append({**self.compute_metrics(lbl, mask, img), **rinfo,
                              "beta": beta})
                if idx == 0:
                    self._save_synth_fig(img, mask, lbl,
                        np.full_like(lbl, -1),   # placeholder para GC
                        os.path.join(out, f"icm_beta{beta}_img0.png"),
                        title=f"Approach 3 ICM | {self.A3_GROUP} | beta={beta}")
            self.results_a3_icm[beta] = rlist
            iou = np.mean([r["mean_iou"] for r in rlist])
            print(f"IoU={iou:.4f}")

        # Barrido GC
        print(f"   GC sweep (n_gaussians={self.A3_NGAUSS_GC})...")
        for beta in self.BETA_GRID:
            print(f"     beta={beta:.1f}", end=" ", flush=True)
            rlist = []
            for idx, (img, mask) in enumerate(zip(imgs, masks)):
                lbl, rinfo = self._run_gc(img, beta, self.A3_NGAUSS_GC)
                rlist.append({**self.compute_metrics(lbl, mask, img), **rinfo,
                              "beta": beta})
                if idx == 0:
                    self._save_synth_fig(img, mask,
                        np.full_like(lbl, -1),   # placeholder para ICM
                        lbl,
                        os.path.join(out, f"gc_beta{beta}_img0.png"),
                        title=f"Approach 3 GC | {self.A3_GROUP} | beta={beta}")
            self.results_a3_gc[beta] = rlist
            iou = np.mean([r["mean_iou"] for r in rlist])
            print(f"IoU={iou:.4f}")

        self._save_approach3_figures(os.path.join(self.out_dir, "figures"))
        self._save_a3_summary(out)

    def save_figures(self):
        print("[6/6] Figuras guardadas.")

    def run(self):
        self.generate_database()
        self.load_real_images()
        self.run_approach1()
        self.run_approach2()
        self.run_approach3()
        self.save_figures()
        print(f"\nHecho. Resultados en: {self.out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    REAL_FOLDER = r"C:\Users\user\Desktop\TFG-Teleco\New_Tool\Imagenes_test"
    OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2")

    comp = BenchmarkComparator(real_folder=REAL_FOLDER, out_dir=OUT_DIR)
    comp.run()

    print("\n-- Approach 1 ------------------------------------------")
    for gname in ["G1_SNR40", "G5_SNR1"]:
        print(f"  {gname}:")
        for method, key in [("ICM", "icm"), ("GC", "gc")]:
            vals = comp.results_a1[gname][key]
            acc  = np.mean([r["accuracy"] for r in vals])
            iou  = np.mean([r["mean_iou"] for r in vals])
            t    = np.mean([r["time_s"]   for r in vals])
            print(f"    {method}: acc={acc:.3f}  IoU={iou:.3f}  t={t:.2f}s")

    print("\n-- Approach 2 ------------------------------------------")
    for gname in ["G1_SNR40", "G5_SNR1"]:
        print(f"  {gname}:")
        for method, key in [("ICM", "icm"), ("GC", "gc")]:
            vals = comp.results_a2[gname][key]
            acc  = np.mean([r["accuracy"] for r in vals])
            iou  = np.mean([r["mean_iou"] for r in vals])
            t    = np.mean([r["time_s"]   for r in vals])
            print(f"    {method}: acc={acc:.3f}  IoU={iou:.3f}  t={t:.2f}s")

    print("\n-- Approach 3 (beta sweep en G3_SNR6) ------------------")
    for method_label, results in [("ICM", comp.results_a3_icm),
                                   ("GC",  comp.results_a3_gc)]:
        print(f"  {method_label}:")
        for beta, rlist in results.items():
            iou = np.mean([r["mean_iou"] for r in rlist])
            t   = np.mean([r["time_s"]   for r in rlist])
            print(f"    beta={beta:.1f}: IoU={iou:.3f}  t={t:.2f}s")

    print("\n-- Mostrando figuras -----------------------------------")
    matplotlib.use('TkAgg')
    figs_dir = os.path.join(OUT_DIR, "figures")
    for fname in sorted(os.listdir(figs_dir)):
        if fname.endswith(".png"):
            path = os.path.join(figs_dir, fname)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(mpimg.imread(path))
            ax.axis('off')
            ax.set_title(fname.replace(".png", ""), fontsize=9)
            plt.tight_layout()
            plt.show()