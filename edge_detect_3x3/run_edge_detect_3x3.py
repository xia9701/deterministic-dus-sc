from __future__ import annotations
import os, math, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from kernels_3x3 import get_kernels_3x3
from utils import unipolar_to_bipolar, bipolar_to_unipolar, replicate_expand, rotate_expand
from streams import BitstreamGeneratorPair
from sc_ops import F_mul, uSADD_sum_value_from_streams
from vlsi22 import vlsi22_pair_streams
from temporal_unary import conv3x3_temporal_unary_counts

# reproducibility
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def method_label(m: str) -> str:
    mapping = {
        "adus_sdus": "DUS (proposed)",
        "sobol": "Sobol",
        "halton": "Halton1_Halton2",
        "lfsr": "LFSR",
        "random8": "Random8_1_Random8_2",
        "umul": "uGEMM‘2020",
        "htc": "HTC‘2025",
        "vlsi22": "Downscale‘2022",
        "tu": "TU_TemporalUnary",
        "tub2": "TubGEMM‘2023",
    }
    return mapping.get(m.lower(), m.upper())

def evaluate_methods_on_image(img_path: str,
                              methods=("adus_sdus","sobol","halton","lfsr","random8","umul","htc","vlsi22","tu","tub2"),
                              lengthN: int = 256,
                              invert_output: bool = True,
                              save_dir: str = "./SC_Outputs",
                              fresh_for=("sobol","halton","lfsr","random8","umul")):
    os.makedirs(save_dir, exist_ok=True)

    # load image
    img_rgb = Image.open(img_path).convert("RGB")
    orig_color = np.array(img_rgb)
    gray = np.array(img_rgb.convert("L"), dtype=float) / 255.0
    H, W = gray.shape

    # kernels + padding
    Kx, Ky, norm_factor = get_kernels_3x3("sobel")
    pad = np.pad(gray, ((1,1),(1,1)), mode="reflect")

    # EXACT
    exact_mag = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            gx = 0.0; gy = 0.0
            for mat, assign in ((Kx,"gx"), (Ky,"gy")):
                acc = 0.0
                for m in range(3):
                    for n in range(3):
                        w = float(mat[m, n])
                        if w == 0.0:
                            continue
                        p = float(pad[i+m, j+n])
                        p_bi = unipolar_to_bipolar(p)
                        acc += F_mul(p_bi, w, mode="exact")
                if assign == "gx":
                    gx = acc
                else:
                    gy = acc
            exact_mag[i, j] = math.sqrt(gx*gx + gy*gy)

    exact_mag = (exact_mag / max(1e-12, norm_factor)) * 255.0
    exact_mag = np.clip(exact_mag, 0.0, 255.0)
    exact_disp = (255.0 - exact_mag) if invert_output else exact_mag
    exact_disp = exact_disp.astype(np.uint8)

    exact_path = os.path.join(save_dir, f"conv3x3_EXACT_N{lengthN}.png")
    Image.fromarray(exact_disp).save(exact_path)
    print(f"Saved EXACT baseline: {exact_path}")

    # SC per method
    expand_times = max(1, lengthN // 256)
    results = {}

    for method in methods:
        mlow = method.lower()

        # TU/TUB2 path (count-domain)
        if mlow in ("tu","tub2"):
            tu_mag = conv3x3_temporal_unary_counts(gray, Kx, Ky, lengthN)
            sc_mag = (tu_mag / max(1e-12, norm_factor)) * 255.0
            sc_mag = np.clip(sc_mag, 0.0, 255.0)
            sc_disp = (255.0 - sc_mag) if invert_output else sc_mag
            sc_disp = sc_disp.astype(np.uint8)

            mae = float(np.mean(np.abs(sc_disp.astype(float)/255.0 - exact_disp.astype(float)/255.0)))
            label = method_label(method)
            results[label] = {"mae": mae, "image": sc_disp}

            out_path = os.path.join(save_dir, f"conv3x3_{label}_N{lengthN}_MAE{mae:.4f}.png")
            Image.fromarray(sc_disp).save(out_path)
            print(f"Saved: {out_path} | MAE={mae:.6e}")
            continue

        # bitstream path
        qmc_mode = "fresh" if mlow in fresh_for else "fixed"
        pair = BitstreamGeneratorPair(method, lengthN, qmc_mode=qmc_mode)
        sc_mag = np.zeros((H, W), dtype=float)

        if mlow == "vlsi22":
            q = int(np.sqrt(lengthN))
            assert q*q == lengthN, "VLSI22 requires N to be a perfect square (e.g., 256, 1024)"

        for i in range(H):
            for j in range(W):
                gx = 0.0; gy = 0.0
                for mat, assign in ((Kx,"gx"), (Ky,"gy")):
                    streams = []
                    for m2 in range(3):
                        for n2 in range(3):
                            w = float(mat[m2, n2])
                            if w == 0.0:
                                continue
                            p_prob = float(pad[i+m2, j+n2])
                            w_prob = bipolar_to_unipolar(w)

                            if mlow == "vlsi22":
                                s_pix, s_wgt = vlsi22_pair_streams(p_prob, w_prob, lengthN)
                                s_pix_e = s_pix
                                s_wgt_e = s_wgt
                            else:
                                s_pix = pair.pixel(p_prob)
                                s_pix_e = replicate_expand(s_pix, expand_times)
                                if mlow == "umul":
                                    s_wgt = pair.weight(w_prob, pixel_stream=s_pix)
                                else:
                                    s_wgt = pair.weight(w_prob)
                                s_wgt_e = rotate_expand(s_wgt, expand_times)

                            streams.append(F_mul(s_pix_e, s_wgt_e, mode="xnor"))

                    val = uSADD_sum_value_from_streams(streams, bipolar=True)
                    if assign == "gx":
                        gx = val
                    else:
                        gy = val

                sc_mag[i, j] = math.sqrt(gx*gx + gy*gy)

        sc_mag = (sc_mag / max(1e-12, norm_factor)) * 255.0
        sc_mag = np.clip(sc_mag, 0.0, 255.0)
        sc_disp = (255.0 - sc_mag) if invert_output else sc_mag
        sc_disp = sc_disp.astype(np.uint8)

        mae = float(np.mean(np.abs(sc_disp.astype(float)/255.0 - exact_disp.astype(float)/255.0)))
        label = method_label(method)
        results[label] = {"mae": mae, "image": sc_disp}

        out_path = os.path.join(save_dir, f"conv3x3_{label}_N{lengthN}_MAE{mae:.4f}.png")
        Image.fromarray(sc_disp).save(out_path)
        print(f"Saved: {out_path} | MAE={mae:.6e}")

    # panel (3x3)
    prefer_order = [
        "DUS (proposed)",
        "LFSR",
        "Sobol",
        "uGEMM‘2020",
        "TubGEMM‘2023",
        "Downscale‘2022",
        "HTC‘2025",
    ]
    ordered = [k for k in prefer_order if k in results][:7]
    if len(ordered) < 7:
        extras = [k for k in results.keys() if k not in ordered]
        ordered += extras[:(7 - len(ordered))]

    fig, axes = plt.subplots(3, 3, figsize=(3.5, 3.9), dpi=600)
    axes = axes.ravel()

    axes[0].imshow(orig_color); axes[0].set_title("Original", fontsize=12); axes[0].axis("off")
    axes[1].imshow(exact_disp, cmap="gray"); axes[1].set_title("Exact (FP)", fontsize=12); axes[1].axis("off")

    for idx, name in enumerate(ordered, start=2):
        ax = axes[idx]
        ax.imshow(results[name]["image"], cmap="gray")
        ax.set_title(f"{name}\nMAE={results[name]['mae']:.4f}", fontsize=12)
        ax.axis("off")

    for i in range(2 + len(ordered), 9):
        axes[i].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.18)
    panel_path = os.path.join(save_dir, f"panel3x3_N{lengthN}.png")
    plt.savefig(panel_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved panel: {panel_path}")

    print("\n=== MAE Summary (vs EXACT) ===")
    for name, info in results.items():
        print(f"{name:>22}: MAE = {info['mae']:.6e}")

    return exact_disp, results

if __name__ == "__main__":
    IMG_PATH = "elephant.png"
    LENGTHN = 256
    METHODS = ("adus_sdus","lfsr","sobol","umul","tub2","vlsi22","htc")
    FRESH_FOR = ("sobol","halton","lfsr","random8","umul")
    INVERT = True
    SAVE_DIR = "./SC_Outputs"

    evaluate_methods_on_image(
        IMG_PATH, methods=METHODS, lengthN=LENGTHN,
        invert_output=INVERT, save_dir=SAVE_DIR, fresh_for=FRESH_FOR
    )