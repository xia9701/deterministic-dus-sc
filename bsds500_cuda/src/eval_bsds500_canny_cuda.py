


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SC Roberts edge detection with method pairs:
  - ADUS&SDUS (DUS always fixed)
  - Sobol1&Sobol2
  - Halton1&Halton2
  - LFSR1&LFSR2
  - Random8_1&Random8_2
  - uMUL(Sobol2|A)


"""


from __future__ import annotations

import os
import json
import argparse
from datetime import datetime

# ------------------------------------------------------------
# Runtime switches (MUST be defined before CuPy/SciPy imports)
# ------------------------------------------------------------
# Priority: CLI --use_gpu (in main) will override env USE_GPU,
# but module-level import path needs a default decision here.
# Keep behavior stable: default USE_GPU=1.
USE_GPU = bool(int(os.environ.get("USE_GPU", "1")))

import numpy as np

if USE_GPU:
    import cupy as cp
    xp = cp
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    # ------- CPU wrappers (SciPy) -------
    from scipy.ndimage import gaussian_filter as _gauss_cpu
    from scipy.ndimage import distance_transform_edt as _edt_cpu
    from scipy.signal  import convolve2d as _conv2d_cpu

    def gaussian_filter(arr, sigma=1.0, **kwargs):
        # arr: cp.ndarray -> cp.ndarray
        return cp.asarray(_gauss_cpu(cp.asnumpy(arr), sigma=sigma, **kwargs))

    def distance_transform_edt(arr, **kwargs):
        # arr: cp.ndarray (0/1) -> cp.ndarray (float)
        return cp.asarray(_edt_cpu(cp.asnumpy(arr), **kwargs))

    def conv2d(a, b, mode='same', boundary='fill', fillvalue=0):
        # a,b: cp.ndarray -> cp.ndarray
        return cp.asarray(_conv2d_cpu(cp.asnumpy(a), cp.asnumpy(b),
                                      mode=mode, boundary=boundary, fillvalue=fillvalue))

else:
    xp = np
    # 纯 CPU 直用 SciPy
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from scipy.signal  import convolve2d as conv2d

import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats.qmc import Halton, Sobol
import os, math, random
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib as mpl
mpl.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif']


# ====== Canny + BSDS500  ======
import  glob

from scipy.io import loadmat


# ---------- 1) SC Sobel：返回 gx, gy ----------
def sc_sobel_gxgy(gray01: np.ndarray, method: str, lengthN: int,
                  fresh_for=('sobol','halton','lfsr','random8','umul')) -> tuple[np.ndarray, np.ndarray]:
    """
    gray01: [0,1] float, 2D
    返回：gx_map, gy_map（双极标尺，范围约 [-L1, L1]）
    """
    H, W = gray01.shape
    Kx, Ky, _ = get_kernels_3x3('sobel')
    pad = np.pad(gray01, ((1,1),(1,1)), mode='reflect')

    mlow = method.lower()
    sdus_a = a_dict.get(lengthN)
    qmc_mode = 'fresh' if mlow in fresh_for else 'fixed'
    pair = BitstreamGeneratorPair(method, lengthN, sdus_a=sdus_a, qmc_mode=qmc_mode)

    expand_times = max(1, lengthN // 256)  
    gx_map = np.zeros((H, W), dtype=float)
    gy_map = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            gx = 0.0; gy = 0.0
            for mat, assign in ((Kx,'gx'), (Ky,'gy')):
                streams = []
                for m2 in range(3):
                    for n2 in range(3):
                        w = float(mat[m2, n2])
                        if w == 0.0: 
                            continue
                        p = float(pad[i + m2, j + n2])
                        p_prob = p
                        w_prob = bipolar_to_unipolar(w)

                        if mlow == 'vlsi22':
                            s_pix, s_wgt = vlsi22_pair_streams(p_prob, w_prob, lengthN)
                            s_pix_e = s_pix
                            s_wgt_e = s_wgt
                        else:
                            s_pix = pair.pixel(p_prob)
                            s_pix_e = replicate_expand(s_pix, expand_times)
                            if mlow == 'umul':
                                s_wgt = pair.weight(w_prob, pixel_stream=s_pix)
                            else:
                                s_wgt = pair.weight(w_prob)
                            s_wgt_e = rotate_expand(s_wgt, expand_times)

                        streams.append(F_mul(s_pix_e, s_wgt_e, mode='xnor'))

                val = uSADD_sum_value_from_streams(streams, bipolar=True)
                if assign == 'gx': gx = val
                else:               gy = val

            gx_map[i, j] = gx
            gy_map[i, j] = gy

    return gx_map, gy_map
def sc_sobel_gxgy_gpu(gray01: cp.ndarray, method: str, lengthN: int,
                      fresh_for=('sobol','halton','lfsr','random8','umul'),
                      tile_h: int = 256, tile_w: int = 256) -> tuple[cp.ndarray, cp.ndarray]:
    """
    与 CPU 版 sc_sobel_gxgy 一致的数值路径，但把 3x3/位流/XNOR/uSADD 全部批量化到 GPU。
    采用 spatial tiling 限制峰值显存，默认 256x256。
    """
    H, W = gray01.shape
    Kx, Ky, _ = get_kernels_3x3('sobel')
    Kx = cp.asarray(Kx); Ky = cp.asarray(Ky)

    pad = cp.pad(gray01, ((1,1),(1,1)), mode='reflect')

    mlow = method.lower()
    sdus_a = a_dict.get(lengthN)
    qmc_mode = 'fresh' if mlow in fresh_for else 'fixed'

    # 预建 pair（沿用生成逻辑：只是在 GPU 上用）
    pair = BitstreamGeneratorPair(method, lengthN, sdus_a=sdus_a, qmc_mode=qmc_mode)

    expand_times = max(1, lengthN // 256)
    gx_map = cp.zeros((H, W), dtype=cp.float32)
    gy_map = cp.zeros((H, W), dtype=cp.float32)

    # 准备右侧阈值（lengthN）常量到 GPU（用于批比较）
    # 注意：BitstreamGeneratorPair 内部用的是 NumPy；这里我们只把输出位流在GPU组装
    A_cache = {}

    def _vlsi22_streams_gpu(p_prob: cp.ndarray, w_prob: float, N: int):
        """
        GPU 向量化生成 VLSI'22 的成对位流：
          - p_prob: (Th,Tw) 像素概率
          - w_prob: 标量权值概率（Sobel/2 后只有 {0,0.5,1} 三种幅度，符号由 XNOR 处理）
          - 返回 (bs_pix, bs_wgt): uint8, 形状 (Th,Tw,N)
        逻辑：k=round(p*P)，下采样到 q=√P 的 TB，展开到 P，再做 A/B 两侧误差补偿（均匀取样）。
        """
        P = int(N)
        q = int(cp.sqrt(P).item())
        assert q*q == P, "VLSI'22 需要 lengthN 为完全平方（如 256）"
    
        Th, Tw = p_prob.shape
        nPix = Th * Tw
    
        # ===== A 路（像素）下采样到 q-bit TB =====
        kA  = cp.rint(p_prob * P).astype(cp.int32)       # (Th,Tw) in [0..P]
        Aq  = kA // q                                    # 欠近似行数 0..q
        eA  = kA - Aq * q                                # 误差 0..q-1
    
        Aqf = Aq.reshape(-1)                              # (nPix,)
        eAf = eA.reshape(-1)                              # (nPix,)
    
        cols = cp.arange(q, dtype=cp.int32)[None, :]      # (1,q)
        tbA  = (cols < Aqf[:, None]).astype(cp.uint8)     # (nPix, q) 行热码
        sA   = cp.repeat(tbA, q, axis=1)                  # (nPix, P) 行重复展开
    
        # ===== B 路（权值）下采样到 q-bit TB（标量）=====
        kB = int(round(float(w_prob) * P))
        Bq = kB // q
        eB = kB - Bq * q
    
        tbB = cp.zeros((q,), dtype=cp.uint8)
        if Bq > 0:
            tbB[:Bq] = 1
        sB_col = cp.tile(tbB, q)                          # (P,) 列平铺展开
        sB = cp.broadcast_to(sB_col[None, :], (nPix, P)).copy()  # (nPix,P) 可写
    
        # ===== A 侧误差补偿：在行 r0=Aq 上翻 invA = round(eA * (Bq/q)) 个 0→1，优先均匀分布 =====
        invA = cp.rint(eAf.astype(cp.float32) * (float(Bq) / float(q))).astype(cp.int32)  # (nPix,)
        invA_clamp = cp.clip(invA, 0, q)                         # 防越界
        baseA = (Aqf * q).astype(cp.int32)                       # 每像素“首个 0 行”的行首线性位置（长度 P 的索引）
    
        j = cp.arange(q, dtype=cp.int32)[None, :]                # (1,q)
        # 选取等距列：pos = floor(j * q / invA)（只在 j<invA 时有效）
        invA_safe = cp.maximum(invA_clamp, 1)[:, None]           # (nPix,1)
        posA = (j * q) // invA_safe                              # (nPix,q)
        useA = (j < invA_clamp[:, None])                         # (nPix,q)
    
        row_off = (cp.arange(nPix, dtype=cp.int32) * P)[:, None] # (nPix,1)
        flat_idxA = row_off + baseA[:, None] + posA              # (nPix,q)
        sA = sA.reshape(-1)
        sA[flat_idxA[useA]] = 1
        sA = sA.reshape(nPix, P)
    
        # ===== B 侧误差补偿：在列 c0=Bq 的 q 个行槽位里翻 invB = round(eB * (Aq/q)) 个 0→1 =====
        invB = cp.rint(float(eB) * (Aqf.astype(cp.float32) / float(q))).astype(cp.int32)
        invB_clamp = cp.clip(invB, 0, q)
        invB_safe  = cp.maximum(invB_clamp, 1)[:, None]          # (nPix,1)
    
        rpos = (j * q) // invB_safe                              # (nPix,q)
        useB = (j < invB_clamp[:, None])                         # (nPix,q)
        colB = (rpos * q + int(Bq)).astype(cp.int32)             # (nPix,q), 每像素列内线性位置
        flat_idxB = row_off + colB
        sB = sB.reshape(-1)
        sB[flat_idxB[useB]] = 1
        sB = sB.reshape(nPix, P)
    
        # reshape 回 (Th,Tw,P)
        return sA.reshape(Th, Tw, P).astype(cp.uint8), sB.reshape(Th, Tw, P).astype(cp.uint8)

    def _bitstream_from_prob_map(prob_map, A_np, N):
        Th, Tw = prob_map.shape
        val8 = (cp.rint(prob_map * 255.0)).astype(cp.int32)       # 0..255
        num  = val8 * (N - 1)
        rhs  = (cp.asarray(A_np) * 255).astype(cp.int32)[None,None,:]  # (1,1,N)
        bs = (num[:, :, None] >= rhs).astype(cp.uint8)
    
        # 饱和：p==0 → 全0；p==255 → 全1
        m0   = (val8 == 0)
        m255 = (val8 == 255)
        if m0.any():
            bs = cp.where(m0[:, :, None], 0, bs)
        if m255.any():
            bs = cp.where(m255[:, :, None], 1, bs)
        return bs

    # 避免一次展开9个邻域×两路×N×H×W爆显存，做空间切片：
    for y0 in range(0, H, tile_h):
        for x0 in range(0, W, tile_w):
            y1 = min(H, y0 + tile_h)
            x1 = min(W, x0 + tile_w)

            gx_tile = cp.zeros((y1 - y0, x1 - x0), dtype=cp.float32)
            gy_tile = cp.zeros_like(gx_tile)

            # 对两个方向各跑一遍（与原逻辑一致）
            for mat, assign in ((Kx,'gx'), (Ky,'gy')):
                streams_vals = []  # 每个非零核权重对应一条“乘积位流的均值（双极）”，最后走 uSADD_sum 的等价式

                for m2 in range(3):
                    for n2 in range(3):
                        used_direct_contrib = False
                        w = float(mat[m2, n2].item())
                        if abs(w) < 1e-12:
                            continue

                        # 取对应邻域 patch
                        patch = pad[y0 + m2:y1 + m2, x0 + n2:x1 + n2]  # (Th, Tw)
                        p_prob = patch
                        w_prob = (w + 1.0) / 2.0

                        if mlow == 'vlsi22':

                            s_pix_e, s_wgt_e = _vlsi22_streams_gpu(p_prob, w_prob, lengthN)


                        elif mlow == 'htc':
                            # ==== NEW: HTC 专用 GPU 向量化生成 ====
                            # 1) pixel: TB（前 ones 位为 1）
                            #    ones = round(p * N)，用广播构造 (Th,Tw,N) 的位流
                            val8 = cp.rint(p_prob * 255.0).astype(cp.int32)
                            ones = cp.floor(val8 * lengthN / 255.0).astype(cp.int32)  # 0..N
                            idx  = cp.arange(lengthN, dtype=cp.int32)[None, None, :]  # (1,1,N)
                            bs_pix = (idx < ones[:, :, None]).astype(cp.uint8)        # (Th,Tw,N)
            
                            # 2) weight: RB by FSM（对当前权值是常量 → 先在 CPU 生成一条，再广播）
                            rb_np = htc_encode_RB_by_FSM(w_prob, lengthN, pair._htc_sel)  # np.ndarray (N,)
                            bs_wgt = cp.asarray(rb_np, dtype=cp.uint8)[None, None, :].repeat(bs_pix.shape[0], axis=0).repeat(bs_pix.shape[1], axis=1)
            
                            s_pix_e = bs_pix
                            s_wgt_e = bs_wgt
                        elif mlow == 'tub2':
                            # 1) 双极像素 x
                            x = 2.0 * p_prob - 1.0  # (Th, Tw)
                    
                            # 2) 分解正/负部分（都在 [0,1]）
                            xp = cp.maximum(x, 0.0)
                            xn = cp.maximum(-x, 0.0)
                    
                            # 3) 权值分解（标量）
                            wp = max(w, 0.0)
                            wn = max(-w, 0.0)
                    
                            # 4) 量化到计数（0..N）
                            Nq  = int(lengthN)
                            Kxp = cp.rint(xp * Nq).astype(cp.int32)   # (Th,Tw)
                            Kxn = cp.rint(xn * Nq).astype(cp.int32)   # (Th,Tw)
                            Kwp = int(round(wp * Nq))                 # 标量
                            Kwn = int(round(wn * Nq))                 # 标量
                    
                            # 5) 计数域的“单极乘法”等价：floor(K_a * K_b / N)
                            #    四象限组合后，回到双极： (Kpp+Knn)-(Kpn+Knp)
                            Kpp = (Kxp * Kwp) // Nq
                            Knn = (Kxn * Kwn) // Nq
                            Kpn = (Kxp * Kwn) // Nq
                            Knp = (Kxn * Kwp) // Nq
                    
                            Kprod_bip = (Kpp + Knn) - (Kpn + Knp)  # ∈ [-N, N]
                    
                            # 6) 计数→双极值（当前 tap 对应的贡献）
                            contrib = Kprod_bip.astype(cp.float32) / float(Nq)  # (Th,Tw)
                    
                            streams_vals.append(contrib)

                            used_direct_contrib = True
                        else:
                            # 其余方法：使用 pair.pixel / pair.weight 生成 **阈值数组A**，然后一次性广播比较
                            # 缓存生成的 A，避免重复构造
                            key_pix = (mlow, 'pix', lengthN)
                            key_wgt = (mlow, 'wgt', lengthN)
                            if key_pix not in A_cache or key_wgt not in A_cache or qmc_mode=='fresh':

                                # 若 fresh，则每次都需重抽；这与原始行为一致
                                # 这里用 method-specific 的阈值生成器重现 A：
                                if mlow == 'adus_sdus':
                                    A_cache[key_pix] = BitstreamGenerator('adus', lengthN).A
                                    A_cache[key_wgt] = BitstreamGenerator('sdus', lengthN, sdus_a=sdus_a).A
                                elif mlow == 'sobol':
                                    A_cache[key_pix], A_cache[key_wgt] = sobol_thresholds_pair(lengthN)
                                elif mlow == 'halton':
                                    A_cache[key_pix], A_cache[key_wgt] = halton_thresholds_pair(lengthN)
                                elif mlow == 'lfsr':
                                    A_cache[key_pix] = generate_lfsr_sequence_auto(0xA5, lengthN)
                                    A_cache[key_wgt] = generate_lfsr_sequence_auto(0xC3, lengthN)
                                elif mlow == 'random8':
                                    A_cache[key_pix], A_cache[key_wgt] = random8_thresholds_pair(lengthN)
                                elif mlow == 'umul':
                                    # uMUL 的权值阈值取决于像素位；为了保持严格同源，仍调用 pair.weight
                                    pass
                                else:
                                    pass


                            
                            if mlow != 'umul':
                                bs_pix = _bitstream_from_prob_map(p_prob, A_cache[key_pix], lengthN)
                                bs_wgt = _bitstream_from_prob_map(cp.full_like(p_prob, w_prob), A_cache[key_wgt], lengthN)
                            else:
                                # ===== uMUL：权值阈值跟像素位逐位选择 =====
                                val8w = int(round(w_prob * 255.0))
                                if val8w <= 0:
                                    # w==0 → 全0（无须看像素位）
                                    bs_pix = _bitstream_from_prob_map(p_prob, sobol_thresholds_pair(lengthN)[0], lengthN)
                                    bs_wgt = cp.zeros_like(bs_pix)
                                elif val8w >= 255:
                                    # w==1 → 全1（无须看像素位）
                                    bs_pix = _bitstream_from_prob_map(p_prob, sobol_thresholds_pair(lengthN)[0], lengthN)
                                    bs_wgt = cp.ones_like(bs_pix)
                                else:
                                    numw = val8w * (lengthN - 1)
                            
                                    if qmc_mode == 'fixed':
                                        # fixed：每个 tile/方向只生成一次，仍然向量化
                                        if ('umul_pack_fixed' not in A_cache):
                                            A_pix_np, A_one_np = sobol_thresholds_pair(lengthN)
                                            B1, _ = sobol_thresholds_pair(lengthN)
                                            # 防“重复”：给 A_zero 加一个相位滚动
                                            A_zero_np = np.roll(B1, 1)
                                            A_cache['umul_pack_fixed'] = (
                                                A_pix_np,
                                                (cp.asarray(A_one_np,  dtype=cp.int32) * 255)[None, None, :],
                                                (cp.asarray(A_zero_np, dtype=cp.int32) * 255)[None, None, :],
                                            )
                                        A_pix_np, rhs1, rhs0 = A_cache['umul_pack_fixed']
                            
                                        bs_pix = _bitstream_from_prob_map(p_prob, A_pix_np, lengthN)
                                        bs_wgt = cp.where(bs_pix == 1,
                                                          (numw >= rhs1).astype(cp.uint8),
                                                          (numw >= rhs0).astype(cp.uint8))
                                    else:
                                        # fresh：**每个 3×3 tap 都重新抽三套阈值**（像 CPU 版语义）
                                        A_pix_np, A_one_np  = sobol_thresholds_pair_fresh(lengthN)
                                        A_zero_np, _        = sobol_thresholds_pair_fresh(lengthN)
                            
                                        rhs1 = (cp.asarray(A_one_np,  dtype=cp.int32) * 255)[None, None, :]
                                        rhs0 = (cp.asarray(A_zero_np, dtype=cp.int32) * 255)[None, None, :]
                            
                                        bs_pix = _bitstream_from_prob_map(p_prob, A_pix_np, lengthN)
                                        bs_wgt = cp.where(bs_pix == 1,
                                                          (numw >= rhs1).astype(cp.uint8),
                                                          (numw >= rhs0).astype(cp.uint8))

                            # replicate / rotate
                            if expand_times > 1:
                                bs_pix = cp.tile(bs_pix, (1,1,expand_times))
                                # rotate-expand 权值流
                                L = lengthN
                                tiled = cp.empty((bs_wgt.shape[0], bs_wgt.shape[1], L*expand_times), dtype=cp.uint8)
                                for ii in range(expand_times):
                                    tiled[:,:,ii*L:(ii+1)*L] = cp.roll(bs_wgt, -ii, axis=2)
                                bs_wgt = tiled

                            s_pix_e = bs_pix
                            s_wgt_e = bs_wgt

                        # XNOR
                        if not used_direct_contrib:
                            prod = cp.logical_not(cp.logical_xor(s_pix_e, s_wgt_e)).astype(cp.uint8)  # (Th,Tw,L')
                        # 计算该“流”的单极均值，再转双极并乘以 K（用 uSADD 等价式）
                            ones_total = prod.sum(axis=2)                           # (Th,Tw)
                            Lp = prod.shape[2]
                            p_mean = ones_total / Lp
                            x_mean = 2.0 * p_mean - 1.0
                            streams_vals.append(x_mean)  # 先存起来

                # uSADD_sum_value_from_streams 等价式：K * mean（双极）
                if len(streams_vals) == 0:
                    val_map = cp.zeros_like(gx_tile)
                else:
                    K = len(streams_vals)
                    val_map = K * cp.stack(streams_vals, axis=0).mean(axis=0)  # (Th, Tw)

                if assign == 'gx':
                    gx_tile = val_map
                else:
                    gy_tile = val_map

            gx_map[y0:y1, x0:x1] = gx_tile
            gy_map[y0:y1, x0:x1] = gy_tile

    return gx_map, gy_map


# ---------- 2) FP Sobel（精确基线） ----------
def fp_sobel_gxgy(gray01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Kx, Ky, _ = get_kernels_3x3('sobel')
    H, W = gray01.shape
    pad = np.pad(gray01, ((1,1),(1,1)), mode='reflect')
    gx = np.zeros((H, W), dtype=float)
    gy = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            v = pad[i:i+3, j:j+3]
            gx[i, j] = np.sum((2*v-1) * Kx)  # 双极
            gy[i, j] = np.sum((2*v-1) * Ky)
    return gx, gy


# ---------- 3) Canny 模块：NMS + 双阈值 + 滞后 ----------
def gradient_mag_dir(gx: np.ndarray, gy: np.ndarray):
    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # [0,180)
    return mag, ang

def nms_suppress(mag, ang_deg):
    xp = cp.get_array_module(mag) if USE_GPU else np
    H, W = mag.shape
    ang = ang_deg * (xp.pi / 180.0)
    gx = xp.cos(ang); gy = xp.sin(ang)

    def bilinear_sample(img, y, x):
        y0 = xp.floor(y).astype(xp.int32); x0 = xp.floor(x).astype(xp.int32)
        y1 = xp.clip(y0 + 1, 0, H-1);      x1 = xp.clip(x0 + 1, 0, W-1)
        wy = y - y0; wx = x - x0
        Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
        return (Ia*(1-wx)*(1-wy) + Ib*wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)

    # 用 reflect pad 取 1 像素外的采样坐标
    yy, xx = xp.meshgrid(xp.arange(H), xp.arange(W), indexing='ij')
    y_f = yy.astype(xp.float32); x_f = xx.astype(xp.float32)
    # 前后各半步（0.5 像素）更接近标准实现；也可用 1.0
    v1 = bilinear_sample(mag, xp.clip(y_f + 0.5*gy, 0, H-1), xp.clip(x_f + 0.5*gx, 0, W-1))
    v2 = bilinear_sample(mag, xp.clip(y_f - 0.5*gy, 0, H-1), xp.clip(x_f - 0.5*gx, 0, W-1))
    keep = (mag >= v1) & (mag >= v2)

    out = xp.zeros_like(mag)
    out[keep] = mag[keep]
    out[0,:]=out[-1,:]=0; out[:,0]=out[:,-1]=0
    return out

def hysteresis(nms255, high: float, low: float) -> np.ndarray:
    xp_local = cp if USE_GPU else np
    H, W = nms255.shape
    strong = (nms255 >= high)
    weak   = (nms255 >= low) & ~strong

    out = xp_local.zeros((H, W), dtype=xp_local.uint8)
    out[strong] = 1

    # 8邻域结构元
    selem = xp_local.array([[1,1,1],[1,1,1],[1,1,1]], dtype=xp_local.uint8)

    # 用二值膨胀替代 BFS：每次把 strong 膨胀后与 weak 交集促升
    # cupyx 没有直接的 binary_dilation，这里用卷积近似（阈值>0）


    changed = True
    while changed:
        if USE_GPU:
            neigh = (conv2d(out, selem, mode='same', boundary='fill', fillvalue=0) > 0)
        else:
            neigh = (conv2d(out, selem, mode='same', boundary='fill', fillvalue=0) > 0)

        to_promote = neigh & weak
        num_new = int(to_promote.sum().get() if USE_GPU else to_promote.sum())
        if num_new == 0:
            changed = False
        else:
            out[to_promote] = 1
            weak[to_promote] = 0

    if USE_GPU:
        return cp.asnumpy(out.astype(cp.uint8))
    else:
        return out.astype(np.uint8)




# ---------- 4) BSDS500 载入与 GT 边界合成 ----------
def load_bsds_split(bsds_root: str, split: str='test'):
    """
    兼容以下结构：
      <root>/BSDS500/data/images/{train,val,test}/*.jpg
      <root>/BSDS500/data/images/{training,validation,test}/*.jpg
    以及 <root> 本身就是 BSDS500 的情况。
    """
    split_alias = {
        'train': ['train', 'training'],
        'val':   ['val', 'validation'],
        'test':  ['test']
    }
    candidates = []
    # 情况 A：root 下有 BSDS500/
    for sp in split_alias.get(split, [split]):
        img_dir = os.path.join(bsds_root, 'BSDS500', 'data', 'images', sp)
        gt_dir  = os.path.join(bsds_root, 'BSDS500', 'data', 'groundTruth', sp)
        candidates.append((img_dir, gt_dir))
    # 情况 B：root 本身就是 BSDS500/
    for sp in split_alias.get(split, [split]):
        img_dir = os.path.join(bsds_root, 'data', 'images', sp)
        gt_dir  = os.path.join(bsds_root, 'data', 'groundTruth', sp)
        candidates.append((img_dir, gt_dir))

    pairs = []
    picked = None
    for img_dir, gt_dir in candidates:
        if os.path.isdir(img_dir) and os.path.isdir(gt_dir):
            img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
            if img_list:
                for ip in img_list:
                    base = os.path.splitext(os.path.basename(ip))[0]
                    mp = os.path.join(gt_dir, base + '.mat')
                    if os.path.exists(mp):
                        pairs.append((ip, mp))
                picked = (img_dir, gt_dir)
                break

    if not pairs:
        raise FileNotFoundError(
            f"[BSDS] No image/GT pairs found for split='{split}'.\n"
            f"Checked dirs:\n" +
            "\n".join([f"  images={i}, groundTruth={g}" for i,g in candidates]) +
            f"\nPlease set BSDS_ROOT correctly (it should contain 'BSDS500/')."
        )
    else:
        print(f"[BSDS] Using images dir: {picked[0]}")
        print(f"[BSDS] Using GT dir    : {picked[1]}")
        print(f"[BSDS] Found {len(pairs)} image/GT pairs.")
    return pairs



def gt_probability_boundary(gt_mat_path: str) -> np.ndarray:
    """
    把多标注者边界合成为概率边界图（0..1）：各 annotator 的 boundary 取均值。
    兼容不同 SciPy loadmat 行为（dict / mat_struct / list / ndarray-of-objects）。
    """
    try:
        # SciPy 1.7+ 推荐：直接把 cell/struct 展平成 Python 原生对象
        mat = loadmat(gt_mat_path, simplify_cells=True)
        gt = mat['groundTruth']               # -> list[dict] 或 dict
        gt_list = gt if isinstance(gt, list) else [gt]
        acc = None; cnt = 0
        for entry in gt_list:
            # entry 是 dict: {'Boundaries': array, 'Segmentation': array, ...}
            bmap = np.asarray(entry['Boundaries'], dtype=float)  # 0/1
            acc = bmap if acc is None else (acc + bmap)
            cnt += 1
        return acc / max(1, cnt)

    except TypeError:
        # 旧式回退：squeeze 后得到 mat_struct / ndarray-of-objects
        mat = loadmat(gt_mat_path, squeeze_me=True, struct_as_record=False)
        gt = mat['groundTruth']
        # 统一为 Python 列表
        if isinstance(gt, np.ndarray):
            gt_list = gt.flatten().tolist()
        else:
            gt_list = [gt]
        acc = None; cnt = 0
        for entry in gt_list:
            # entry 可能是 mat_struct 或 dict
            if hasattr(entry, 'Boundaries'):
                bmap = np.asarray(entry.Boundaries, dtype=float)
            elif isinstance(entry, dict) and 'Boundaries' in entry:
                bmap = np.asarray(entry['Boundaries'], dtype=float)
            else:
                raise KeyError("Cannot find 'Boundaries' in groundTruth entry")
            acc = bmap if acc is None else (acc + bmap)
            cnt += 1
        return acc / max(1, cnt)

def boundary_pr(gt_pb: np.ndarray, pred_edge: np.ndarray, tol_frac: float=0.0075):
    H, W = gt_pb.shape
    diag = math.hypot(H, W)
    rho  = max(1.0, tol_frac * diag)

    gt_bin = (gt_pb > 0).astype(np.uint8)
    pr_bin = (pred_edge > 0).astype(np.uint8)

    dt_gt = distance_transform_edt(1 - gt_bin)
    match_pred = (pr_bin == 1) & (dt_gt <= rho)
    TP_prec = int(match_pred.sum())
    P_pred  = int(pr_bin.sum())

    dt_pr = distance_transform_edt(1 - pr_bin)
    match_gt = (gt_bin == 1) & (dt_pr <= rho)
    TP_rec = int(match_gt.sum())
    P_gt   = int(gt_bin.sum())

    TP = min(TP_prec, TP_rec)
    FP = max(0, P_pred - TP)
    FN = max(0, P_gt   - TP)
    return TP, FP, FN


# ---------- 5) 近似 BSDS 边界评测（距离容忍） ----------
def boundary_pr_once(gt_pb: cp.ndarray, pred_edge: cp.ndarray, tol_frac: float=0.0075):
    # 两者都在 GPU
    H, W = gt_pb.shape
    diag = float(cp.hypot(H, W).get())
    rho  = max(1.0, tol_frac * diag)

    gt_bin = (gt_pb > 0).astype(cp.uint8)
    pr_bin = (pred_edge > 0).astype(cp.uint8)

    dt_gt = distance_transform_edt(1 - gt_bin)  # (H,W) float
    match_pred = (pr_bin == 1) & (dt_gt <= rho)
    TP_prec = int(match_pred.sum().get())
    P_pred  = int(pr_bin.sum().get())

    dt_pr = distance_transform_edt(1 - pr_bin)
    match_gt = (gt_bin == 1) & (dt_pr <= rho)
    TP_rec = int(match_gt.sum().get())
    P_gt   = int(gt_bin.sum().get())

    TP = min(TP_prec, TP_rec)
    FP = max(0, P_pred - TP)
    FN = max(0, P_gt   - TP)
    return TP, FP, FN

# 带缓存的版本（对同一张图反复阈值时复用 dt_gt）
def boundary_pr_cached(gt_pb_gpu: cp.ndarray, pred_edge: cp.ndarray, tol_frac: float, dt_gt_cache: dict):
    H, W = gt_pb_gpu.shape
    key = int(gt_pb_gpu.data.ptr)  # 以设备指针作为 key（简单有效）
    if key not in dt_gt_cache:
        gt_bin = (gt_pb_gpu > 0).astype(cp.uint8)
        dt_gt_cache[key] = distance_transform_edt(1 - gt_bin)  # 缓存
    rho  = max(1.0, tol_frac * float(cp.hypot(H, W).get()))
    pr_bin = (pred_edge > 0).astype(cp.uint8)
    TP_prec = int(((pr_bin == 1) & (dt_gt_cache[key] <= rho)).sum().get())
    P_pred  = int(pr_bin.sum().get())

    dt_pr = distance_transform_edt(1 - pr_bin)
    gt_bin = (gt_pb_gpu > 0).astype(cp.uint8)
    TP_rec = int(((gt_bin == 1) & (dt_pr <= rho)).sum().get())
    P_gt   = int(gt_bin.sum().get())

    TP = min(TP_prec, TP_rec)
    FP = max(0, P_pred - TP)
    FN = max(0, P_gt   - TP)
    return TP, FP, FN


def precision_recall_f(TP: int, FP: int, FN: int):
    P = TP / (TP + FP + 1e-12)
    R = TP / (TP + FN + 1e-12)
    F = 2*P*R / (P+R+1e-12)
    return P, R, F


# ---------- 6) 整图 Canny（FP 或 SC） ----------
def canny_from_gray(gray01: np.ndarray, mode: str, *, method: str='adus_sdus', lengthN: int=256,
                    gauss_sigma: float=1.0, high_th: float=60.0, low_ratio: float=0.4) -> np.ndarray:
    g = cp.asarray(gray01) if USE_GPU else gray01

    # 1) 高斯
    blur = gaussian_filter(g, sigma=gauss_sigma)

    # 2) Sobel
    if mode == 'fp':
        gx, gy = fp_sobel_gxgy(cp.asnumpy(blur) if USE_GPU else blur)
    else:
        gx, gy = (sc_sobel_gxgy_gpu(blur, method, lengthN) if USE_GPU
                  else sc_sobel_gxgy(blur if isinstance(blur, np.ndarray) else cp.asnumpy(blur), method, lengthN))

    # 3) 幅值与角度
    _, _, norm_factor = get_kernels_3x3('sobel')
    gx = cp.asarray(gx) if USE_GPU and not isinstance(gx, cp.ndarray) else gx
    gy = cp.asarray(gy) if USE_GPU and not isinstance(gy, cp.ndarray) else gy

    mag = (cp.sqrt(gx*gx + gy*gy) if USE_GPU else np.sqrt(gx*gx + gy*gy))
    ang = (cp.rad2deg(cp.arctan2(gy, gx)) % 180.0) if USE_GPU \
          else (np.rad2deg(np.arctan2(gy, gx)) % 180.0)

    # 4) 统一标定到 0..255（用理论上界）
    mag255 = (mag / (norm_factor + 1e-12)) * 255.0
    mag255 = (cp.clip if USE_GPU else np.clip)(mag255, 0.0, 255.0)
    if USE_GPU:
        mag255 = mag255.astype(cp.float32, copy=False)
    else:
        mag255 = mag255.astype(np.float32, copy=False)

    # 5) NMS（在已标定的幅值上）
    nms255 = nms_suppress(mag255, ang)


    # 6) 双阈值 + 滞后
    high = float(high_th)
    low  = float(low_ratio * high)
    edges = hysteresis(nms255, high=high, low=low)

    return edges if isinstance(edges, np.ndarray) else cp.asnumpy(edges)


# ---------- 7) ODS/OIS + PR 曲线 ----------
def evaluate_bsds_canny(bsds_root: str, split: str,
                        mode: str, method: str='adus_sdus', lengthN: int=256,
                        gauss_sigma: float=1.0,
                        high_grid = np.linspace(20, 140, 13),  # 高阈值栅格 (0..255)
                        low_ratio: float=0.4,
                        tol_frac: float=0.0075,
                        max_images: int | None = None,
                        save_plot: str | None = None,
                        save_csv: str | None = None):
    """
    mode: 'fp' 或 'sc'
    返回：{'ODS':(P,R,F,th), 'OIS':(meanF, stdF), 'AP':ap, 'PR': [(P,R,F,th), ...]}
    新增：AP & 可选 PR 数据 CSV 导出（列：th, P, R, F）
    """
    pairs = load_bsds_split(bsds_root, split)
    if len(pairs) == 0:
        raise RuntimeError("No BSDS images found — check BSDS_ROOT and folder names.")
    if max_images is not None:
        pairs = pairs[:max_images]

    # 预取灰度与 GT（并搬到 GPU）
    imgs_gray = []
    gts_pb    = []
    for ip, mp in pairs:
        img = np.array(Image.open(ip).convert('RGB'))
        gray01 = np.dot(img[...,:3],[0.299,0.587,0.114]) / 255.0
        gt_pb = gt_probability_boundary(mp)
        if USE_GPU:
            imgs_gray.append(cp.asarray(gray01, dtype=cp.float32))
            gts_pb.append(cp.asarray(gt_pb, dtype=cp.float32))
        else:
            imgs_gray.append(gray01.astype(float, copy=False))
            gts_pb.append(gt_pb.astype(float, copy=False))



    # ===== ODS =====
    pr_list = []

    dt_cache = {}  # 缓存每张图的 dt_gt
    for th in high_grid:
        TP=FP=FN=0
        for gray01, gt_pb in zip(imgs_gray, gts_pb):
            pred = canny_from_gray((gray01.get() if USE_GPU else gray01), mode,
                                   method=method, lengthN=lengthN,
                                   gauss_sigma=gauss_sigma, high_th=th, low_ratio=low_ratio)
            pred_gpu = cp.asarray(pred) if USE_GPU else pred
            tp, fp, fn = (boundary_pr_cached(gt_pb, pred_gpu, tol_frac, dt_cache) if USE_GPU
                          else boundary_pr(gt_pb, pred, tol_frac))
            TP += tp; FP += fp; FN += fn
        P, R, F = precision_recall_f(TP, FP, FN)
        pr_list.append((P, R, F, float(th)))
        # 选 ODS
    ods_idx = int(np.argmax([x[2] for x in pr_list]))
    ODS = (*pr_list[ods_idx][:3], pr_list[ods_idx][3])

    # ===== OIS =====
    F_per_img = []
    dt_cache = {}  # 重新计一遍缓存（不影响正确性）
    for gray01, gt_pb in zip(imgs_gray, gts_pb):
        bestF = -1.0
        for th in high_grid:
            pred = canny_from_gray((gray01.get() if USE_GPU else gray01), mode,
                                   method=method, lengthN=lengthN,
                                   gauss_sigma=gauss_sigma, high_th=th, low_ratio=low_ratio)
            pred_gpu = cp.asarray(pred) if USE_GPU else pred
            tp, fp, fn = (boundary_pr_cached(gt_pb, pred_gpu, tol_frac, dt_cache) if USE_GPU
                          else boundary_pr(gt_pb, pred, tol_frac))
            _, _, F = precision_recall_f(tp, fp, fn)
            if F > bestF: bestF = F
        F_per_img.append(bestF)
    F_per_img = np.array(F_per_img, float)
    
        
    
    
    OIS = (float(F_per_img.mean()), float(F_per_img.std()))

    # ===== AP：PR 曲线下的面积（按 Recall 升序做梯形积分）=====
    if len(pr_list) >= 2:
        pts = sorted([(r, p) for (p, r, _, _) in pr_list], key=lambda x: x[0])
        ap = 0.0
        prev_r, prev_p = pts[0]
        for r, p in pts[1:]:
            ap += 0.5 * (p + prev_p) * max(0.0, (r - prev_r))
            prev_r, prev_p = r, p
        AP = float(ap)
    else:
        AP = 0.0

    # 可选：保存 PR 曲线 PNG
    if save_plot is not None:
        plt.figure(figsize=(4.0, 4.0), dpi=200)
        Ps = [p for (p,_,_,_) in pr_list]
        Rs = [r for (_,r,_,_) in pr_list]
        plt.plot(Rs, Ps, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR - {mode.upper()} {method}, N={lengthN}\nODS F={ODS[2]:.3f} @ th={ODS[3]:.1f}, OIS F={OIS[0]:.3f}, AP={AP:.3f}')
        plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        plt.close()

    # 可选：保存 PR 原始数据 CSV （th, P, R, F）
    if save_csv is not None:
        import csv
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['th', 'P', 'R', 'F'])
            for (P, R, F, th) in pr_list:
                writer.writerow([f"{th:.6f}", f"{P:.6f}", f"{R:.6f}", f"{F:.6f}"])

    return {'ODS': ODS, 'OIS': OIS, 'AP': AP, 'PR': pr_list}


# =====================
# Reproducibility (仅控制 np.random / random；fresh 的 Sobol/Halton/LFSR/Random8 仍会变化)
# =====================
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =====================
# Helpers: bipolar <-> unipolar
# =====================
def unipolar_to_bipolar(p: float) -> float:
    return 2.0 * p - 1.0

def bipolar_to_unipolar(x: float) -> float:
    return (x + 1.0) / 2.0


# ===== 3x3 kernels (Sobel 默认) =====
def get_kernels_3x3(name: str = 'sobel'):
    """
    返回 (Kx, Ky, norm_factor)
    - 为了与位流双极编码 [-1,1] 对齐，Sobel 权重先除以 2（把 ±2 压到 ±1）
    - norm_factor = sqrt(||Kx||_1^2 + ||Ky||_1^2)，用来把 |G| 归一到 0..255
    """
    name = name.lower()
    if name == 'sobel':
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
        Ky = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=float)
        scale = 2.0   # <<< 把 ±2 压到 ±1
        Kx /= scale; Ky /= scale
    elif name == 'scharr': 
        Kx = np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]], dtype=float)
        Ky = np.array([[ 3, 10,  3],
                       [ 0,  0,  0],
                       [-3,-10, -3]], dtype=float)
        scale = 10.0
        Kx /= scale; Ky /= scale
    else:
        raise ValueError("Unknown kernel set")

    L1x = float(np.abs(Kx).sum())
    L1y = float(np.abs(Ky).sum())
    norm = np.sqrt(L1x**2 + L1y**2)  # <<< |G| <= norm（输入双极幅度≤1）
    return Kx, Ky, norm


# ==== 22‘VLSI helper =====

def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def vlsi22_generate_stream_for_value(p_prob: float, lengthN: int) -> np.ndarray:
    """
    22'VLSI：把长度 P=2^n 的数值 p，按 q=2^{n/2} 下采样并误差补偿，生成一条长度仍为 P 的确定性 GB 流。
    - 下采样：k = round(p*P)，A_q = floor(k/q)，err = k - A_q*q
    - 展开：把 q 位（热码：前 A_q 位为 1）每位重复 q 次 → 长度 P
    - 误差补偿：在“第一位 0”的那一行（索引 = A_q）上，从 q 次重复里**确定性均匀**挑出 err 个位置把 0 翻成 1
    * 注意：要求 P 为完全平方：P= q*q
    """
    P = int(lengthN)
    assert _is_pow2(P), "VLSI22 要求 lengthN=2^n"
    # q 为 sqrt(P)，也需是整数
    q = int(round(np.sqrt(P)))
    assert q * q == P, "VLSI22 需要 lengthN 为完全平方（例如 256=16^2）"

    # 量化到 P
    k = int(round(float(p_prob) * P))
    k = max(0, min(P, k))

    # 下采样到 q
    A_q = k // q                  # under-approx
    err = k - A_q * q             # 0..q-1

    # 构造 q×q 的“行重复”结构（行 i 的值 = 1 if i < A_q else 0）
    out = np.zeros(P, dtype=np.uint8)
    # 先填满前 A_q 行为 1
    if A_q > 0:
        # 行 0..A_q-1 全为 1
        out[:A_q * q] = 1

    # 误差补偿：在“第一位 0”那一行（i = A_q）里，把 err 个 0 均匀翻为 1
    if err > 0 and A_q < q:
        # 在该行的 q 个重复里，挑 err 个均匀分布的位置（确定性：linspace）
        positions = np.linspace(0, q - 1, err, dtype=int)
        row_start = A_q * q
        for pos in positions:
            out[row_start + pos] = 1

    # out 的 1 个数 = A_q*q + err = k，完全匹配量化值
    return out

# ===== HTC helpers: FSM schedule (Sobol-based), RB/TB encoders =====

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def htc_build_fsm_schedule_sobol(lengthN: int) -> np.ndarray:
    """
    用 1D Sobol 的前 lengthN 个数，按论文区间规则生成 FSM 选位表 SEL[k]∈[0..n-1]。
    要求 lengthN=2^n。
    """
    assert _is_power_of_two(lengthN), "HTC 需要 lengthN=2^n"
    n = int(np.log2(lengthN))
    eng = Sobol(d=1, scramble=False, seed=None)
    S = eng.random(n=lengthN)[:, 0]  # [0,1)
    SEL = np.empty(lengthN, dtype=np.int32)

    # 映射规则：找 m∈{1..n} 使 ((2^{m-1}-1)/2^{m-1}) <= S < ((2^m-1)/2^m)，选 x_{n-m}
    for k, Sk in enumerate(S):
        for m in range(1, n + 1):
            left  = (2**(m-1) - 1) / (2**(m-1)) if m > 1 else 0.0  # m=1 左端点 0
            right = (2**m - 1) / (2**m)
            if left <= Sk < right:
                SEL[k] = m - 1
                break
        else:
            # 极少数边界（Sk=1.0不应该出现），兜底到最低位
            SEL[k] = n - 1
    return SEL  # 长度 lengthN，值域 0..n-1（0=LSB, n-1=MSB）

def htc_int_to_bits(x_int: int, n: int) -> np.ndarray:
    """把 0..(2^n-1) 转成 n 位数组（MSB->LSB）。"""
    bits = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        bits[n - 1 - i] = (x_int >> i) & 1  # MSB at index n-1
    return bits

def htc_encode_RB_by_FSM(p_prob: float, lengthN: int, sel: np.ndarray) -> np.ndarray:
    """
    RB：受控均匀（FSM 选位表），输入 p∈[0,1]。
    步骤：把 p 量化为 n 位小数 x_int∈[0..2^n-1]，取 x 的各位，按 SEL 逐拍从某一位输出。
    """
    assert len(sel) == lengthN and _is_power_of_two(lengthN)
    n = int(np.log2(lengthN))

    # p→n位小数：x_int ∈ [0..2^n-1]，p=1 → 全 1（饱和）
    x_int = int(round(p_prob * (2**n)))
    if x_int >= 2**n:  # 饱和
        x_int = 2**n - 1
    if x_int <= 0:
        x_bits = np.zeros(n, dtype=np.uint8)
    else:
        x_bits = htc_int_to_bits(x_int, n)

    out = np.empty(lengthN, dtype=np.uint8)
    for k in range(lengthN):
        bit_idx = sel[k]   # 0..n-1（0=LSB）
        out[k] = x_bits[bit_idx]
    return out

def htc_encode_TB(p_prob: float, lengthN: int) -> np.ndarray:
    """
    TB：时间码（连续 1 段）。硬件感知：移位寄存器把需要的 1 连在一起。
    p→ones_count（整数），前 ones_count 位为 1，剩余为 0。
    """
    # 统一整数域：val8 ∈ [0..255]，再映射到 [0..lengthN]
    val8 = int(round(p_prob * 255.0))
    if val8 <= 0:
        return np.zeros(lengthN, dtype=np.uint8)
    if val8 >= 255:
        return np.ones(lengthN, dtype=np.uint8)
    ones = int(round(val8 * lengthN / 255.0))
    ones = max(0, min(lengthN, ones))
    out = np.zeros(lengthN, dtype=np.uint8)
    if ones > 0:
        out[:ones] = 1  # 固定从 0 起：真实硬件的“epoch起点”
    return out




def _downscale_tb(p_prob: float, P: int):
    """把概率 p 映射到 q=√P 的 TB，并给出欠近似误差 e（单位：bit）。"""
    assert P > 0 and int(math.isqrt(P))**2 == P, "VLSI'22 需 P 为完全平方"
    q = int(math.isqrt(P))
    k = int(round(float(p_prob) * P))           # 0..P
    A_q = k // q                                # 欠近似（向下取整）
    e   = k - A_q * q                           # 0..q-1
    tb  = np.zeros(q, dtype=np.uint8)
    if A_q > 0:
        tb[:A_q] = 1
    return tb, A_q, e, q

def vlsi22_pair_streams(p_prob: float, w_prob: float, P: int):
    """
    VLSI'22：downscale 到 q 位 TB；展开为 P=q^2，并在“首个 0 的那一行/列”做误差补偿。
    关键：A 用行重复（repeat），B 用列平铺（tile），确保 q×q 的配对全覆盖。
    """
    assert int(np.sqrt(P))**2 == P, "VLSI'22 需要 P 为完全平方"
    q = int(np.sqrt(P))

    # downscale 到 q 位 TB：A'、B' 及误差 eA/eB
    def _downscale_tb_local(p):
        k = int(round(float(p) * P))  # 0..P
        A_q = k // q                  # 欠近似（向下）
        e   = k - A_q * q             # 0..q-1
        tb  = np.zeros(q, dtype=np.uint8)
        if A_q > 0:
            tb[:A_q] = 1
        return tb, A_q, e

    tbA, A_q, eA = _downscale_tb_local(p_prob)
    tbB, B_q, eB = _downscale_tb_local(w_prob)

    # 展开到 P：A 行重复，B 列平铺（!!!）
    sA = np.repeat(tbA, q)   # t=r*q+c → tbA[r]
    sB = np.tile(tbB,   q)   # t=r*q+c → tbB[c]

    # A 侧补偿：在行 r0=A_q 的 q 个列槽位里翻 eA * (B_q/q) 个 0→1，优先对齐 B'==1 的列
    if (A_q < q) and (eA > 0):
        invA = int(round(eA * (B_q / q)))
        invA = max(0, min(invA, q))
        if invA > 0:
            cols_one = np.flatnonzero(tbB == 1)     # 列索引集合（B'==1 的列）
            if cols_one.size > 0:
                pick = np.linspace(0, cols_one.size - 1, invA, dtype=int)
                sA[A_q * q + cols_one[pick]] = 1
            else:
                pick = np.linspace(0, q - 1, invA, dtype=int)
                sA[A_q * q + pick] = 1

    # B 侧补偿：在列 c0=B_q 的 q 个行槽位里翻 eB * (A_q/q) 个 0→1，优先对齐 A'==1 的行
    if (B_q < q) and (eB > 0):
        invB = int(round(eB * (A_q / q)))
        invB = max(0, min(invB, q))
        if invB > 0:
            rows_one = np.flatnonzero(tbA == 1)     # 行索引集合（A'==1 的行）
            if rows_one.size > 0:
                pick = np.linspace(0, rows_one.size - 1, invB, dtype=int)
                rsel = rows_one[pick]
                sB[rsel * q + B_q] = 1
            else:
                pick = np.linspace(0, q - 1, invB, dtype=int)
                sB[pick * q + B_q] = 1

    return sA.astype(np.uint8), sB.astype(np.uint8)




# ===== Temporal–Unary helpers (tuGEMM / tubGEMM) =====


def _tu_counts_from_prob(p: float, N: int) -> int:
    """单极概率 p → TB 的 '1' 总数（长度 N）。"""
    return int(round(float(p) * N))

def _tu_mul_bipolar_with_pm1(k_pix: int, w_plus_one: bool, N: int) -> int:
    """
    双极乘法（XNOR）在 ±1 权值下的计数形式：
      +1 → 原样； -1 → 取反
    返回：输出 TB 的 '1' 个数（长度 N 标尺）
    """
    return k_pix if w_plus_one else (N - k_pix)

def _tu_scaled_add_two(k1: int, k2: int) -> int:
    """uSADD 的 2 输入情形：返回 floor((k1+k2)/2)。"""
    return (k1 + k2) // 2

def roberts_temporal_unary(gray: np.ndarray, lengthN: int, *, method_tag: str = 'tu') -> np.ndarray:
    """
    TU/TUB2 路径：在计数域完成 Roberts（乘法无误差，缩放加法用均值）。
    返回 0..255 的 float 图（未反显）。
    """
    H, W = gray.shape
    pad = np.pad(gray, ((0,1),(0,1)), mode='reflect')
    out = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            # Kx: +1 * p(i,j) 与 -1 * p(i+1,j+1)
            k00 = _tu_counts_from_prob(pad[i  , j  ], lengthN)
            k11 = _tu_counts_from_prob(pad[i+1, j+1], lengthN)
            kx1 = _tu_mul_bipolar_with_pm1(k00, True , lengthN)   # +1
            kx2 = _tu_mul_bipolar_with_pm1(k11, False, lengthN)   # -1
            kx  = _tu_scaled_add_two(kx1, kx2)
            gx  = (2.0 * kx / lengthN) - 1.0

            # Ky: +1 * p(i,j+1) 与 -1 * p(i+1,j)
            k01 = _tu_counts_from_prob(pad[i  , j+1], lengthN)
            k10 = _tu_counts_from_prob(pad[i+1, j  ], lengthN)
            ky1 = _tu_mul_bipolar_with_pm1(k01, True , lengthN)
            ky2 = _tu_mul_bipolar_with_pm1(k10, False, lengthN)
            ky  = _tu_scaled_add_two(ky1, ky2)
            gy  = (2.0 * ky / lengthN) - 1.0

            out[i, j] = np.sqrt(gx*gx + gy*gy)

    # 归一化到 0..255（与其他路径一致）
    out = (out / np.sqrt(2.0)) * 255.0
    return out
def conv3x3_temporal_unary_counts(gray: np.ndarray, Kx: np.ndarray, Ky: np.ndarray, N: int) -> np.ndarray:
    """
    通用 3×3 的 TU/TUB2（计数域精确）：支持任意双极权值 w∈[-1,1]。
    - 输入 gray: [0,1] 灰度
    
    - N: unary 长度
    返回：|G| 的未归一化幅度（双极标尺 ∈ [0, sqrt(L1x^2+L1y^2)]），后续再做统一归一化到 0..255
    """
    H, W = gray.shape
    pad = np.pad(gray, ((1,1),(1,1)), mode='reflect')
    out = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            gx_counts = 0  # 累加到计数域（单位：N）
            gy_counts = 0

            for mat, assign in ((Kx, 'gx'), (Ky, 'gy')):
                total_counts = 0
                for m in range(3):
                    for n in range(3):
                        w = float(mat[m, n])
                        if w == 0.0:
                            continue
                        p = float(pad[i + m, j + n])   # [0,1]
                        x = 2.0 * p - 1.0               # 双极 [-1,1]

                        # 正/负部分（都在 [0,1]）
                        xp = max(x, 0.0); xn = max(-x, 0.0)
                        wp = max(w, 0.0); wn = max(-w, 0.0)

                        # 量化到计数（0..N）
                        Kxp = int(round(xp * N)); Kxn = int(round(xn * N))
                        Kwp = int(round(wp * N)); Kwn = int(round(wn * N))

                        # unipolar 乘法的计数结果：floor(K_a * K_b / N)
                        Kpp = (Kxp * Kwp) // N
                        Knn = (Kxn * Kwn) // N
                        Kpn = (Kxp * Kwn) // N
                        Knp = (Kxn * Kwp) // N

                        Kprod_bip = (Kpp + Knn) - (Kpn + Knp)   # ∈ [-N, N]
                        total_counts += Kprod_bip

                if assign == 'gx':
                    gx_counts = total_counts
                else:
                    gy_counts = total_counts

            # 计数→双极值
            gx = gx_counts / float(N)
            gy = gy_counts / float(N)
            out[i, j] = math.sqrt(gx*gx + gy*gy)

    return out

# =====================
# Core bitstream helpers
# =====================
def replicate_expand(bitstream: np.ndarray, times: int) -> np.ndarray:
    return np.tile(bitstream.astype(np.uint8), max(1, times))

def rotate_expand(bitstream: np.ndarray, times: int) -> np.ndarray:
    bitstream = bitstream.astype(np.uint8)
    L = len(bitstream)
    times = max(1, times)
    extended = np.empty(L * times, dtype=np.uint8)
    for i in range(times):
        extended[i*L:(i+1)*L] = np.roll(bitstream, -i)
    return extended

# =====================
# Deterministic Uniform Sequences (DUS)
# =====================
a_dict = {16:7, 32:19, 64:41, 128:83, 256:157, 512:323, 1024:629}

def generate_sdus(N: int, a: int) -> np.ndarray:
    from math import gcd
    assert gcd(a, N) == 1, "a and N must be coprime"
    return np.array([(a * i) % N for i in range(N)], dtype=np.int32)

# =====================
# LFSR sequences (auto bit width)
# =====================
def get_lfsr_taps(bit_width: int):
    primitive_polynomials = {
        4:[4,3], 5:[5,3], 6:[6,5], 7:[7,6], 8:[8,6,5,4], 9:[9,5], 10:[10,7], 11:[11,9],
        12:[12,11,10,4], 13:[13,12,11,8], 14:[14,13,12,2], 15:[15,14], 16:[16,14,13,11]
    }
    if bit_width not in primitive_polynomials:
        raise ValueError(f"No taps defined for bit width {bit_width}")
    return primitive_polynomials[bit_width]
# —— 备选 taps（同位宽给第二个多项式；够用就好，覆盖到常见位宽） ——
_LFSR_TAPS_ALT = {
    4:  [4,1],               # x^4 + x + 1
    5:  [5,2],               # x^5 + x^2 + 1
    6:  [6,1],               # x^6 + x + 1
    7:  [7,1],               # x^7 + x + 1
    8:  [8,6,5,4],           # x^8 + x^6 + x^3 + x^2 + 1
    9:  [9,4],               # x^9 + x^4 + 1
    10: [10,3],              # x^10 + x^3 + 1
    11: [11,2],              # x^11 + x^2 + 1
    12: [12,6,4,1],          # x^12 + x^6 + x^4 + x + 1
    15: [15,1],              # x^15 + x + 1
    16: [16,15,13,4],        # x^16 + x^15 + x^13 + x^4 + 1
    # 如需更大位宽可再补充；如果某位宽没有备选，后面有自动降级策略
}

def get_two_lfsr_taps(bit_width: int) -> tuple[list[int], list[int]]:
    """返回同位宽下两组不同的 taps；若无备选，则第二组返回主 taps（稍后用 decimation 去相关）"""
    primary = get_lfsr_taps(bit_width)
    alt = _LFSR_TAPS_ALT.get(bit_width, primary)
    return primary, alt
def _lfsr_step(sr: int, taps: list[int], mask: int) -> int:
    fb = 0
    for t in taps:
        fb ^= (sr >> (t - 1)) & 1
    sr = ((sr << 1) & mask) | fb
    return sr if sr != 0 else 1  # 避免全零

def generate_lfsr_sequence_auto(seed: int, lengthN: int) -> np.ndarray:
    """
    标准 LFSR-PRNG 基线：用 16-bit 最大长度 LFSR 产生无偏的阈值序列 A∈[0..N-1]。
    仍是 LFSR，只是去掉“sr % N”的结构性伪影，更贴近文献中的做法。
    """
    bit_width = 16
    mask = (1 << bit_width) - 1
    taps = get_lfsr_taps(bit_width)         
    sr = seed & mask
    if sr == 0:
        sr = 1

    vals = np.empty(lengthN, dtype=np.uint32)
    for i in range(lengthN):
        sr = _lfsr_step(sr, taps, mask)
        vals[i] = sr                          # 1..65535 几乎均匀遍历

    # 线性缩放到 [0..N-1]，避免模运算引入的重复图案
    #A = (vals * (lengthN - 1) // mask).astype(np.int32)
    # 以 2^bit_width 为分母做均匀映射；N=256 时等价于取高 8 位
    A = ((vals.astype(np.uint32) * lengthN) >> bit_width).astype(np.int32)

    return A


# =====================
# Random8 thresholds (single & pair)
# =====================
def random8_thresholds(lengthN: int) -> np.ndarray:
    rnd8 = np.random.randint(0, 256, size=lengthN, dtype=np.int32)
    A = (rnd8 * (lengthN - 1) // 255).astype(np.int32)
    return A

def random8_thresholds_pair(lengthN: int) -> tuple[np.ndarray, np.ndarray]:
    return random8_thresholds(lengthN), random8_thresholds(lengthN)

# =====================
# Sobol / Halton thresholds (pair via d=2)
# =====================
def sobol_thresholds_pair(lengthN: int) -> tuple[np.ndarray, np.ndarray]:
    eng = Sobol(d=2, scramble=False, seed=RANDOM_SEED)
    S = eng.random(n=lengthN)
    A1 = (S[:,0] * lengthN).astype(int)
    A2 = (S[:,1] * lengthN).astype(int)
    A1 = np.clip(A1, 0, lengthN-1).astype(np.int32)
    A2 = np.clip(A2, 0, lengthN-1).astype(np.int32)
    return A1, A2
def sobol_thresholds_pair_fresh(lengthN: int) -> tuple[np.ndarray, np.ndarray]:
    """每次调用都新抽一对（fresh）的 Sobol 阈值序列（d=2，seed=None）。"""
    eng = Sobol(d=2, scramble=False, seed=None)  # 不固定种子 → 每次不同
    S = eng.random(n=lengthN)
    A1 = np.clip((S[:, 0] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    A2 = np.clip((S[:, 1] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    return A1, A2

def halton_thresholds_pair(lengthN: int) -> tuple[np.ndarray, np.ndarray]:
    eng = Halton(d=2, scramble=False, seed=RANDOM_SEED)
    H = eng.random(n=lengthN)
    A1 = (H[:,0] * lengthN).astype(int)
    A2 = (H[:,1] * lengthN).astype(int)
    A1 = np.clip(A1, 0, lengthN-1).astype(np.int32)
    A2 = np.clip(A2, 0, lengthN-1).astype(np.int32)
    return A1, A2





# =====================
# Single-lane generator (with optional A override)
# =====================
class BitstreamGenerator:
    """Create one threshold array A and map p∈[0,1] → bitstream via integer-domain >= compare + saturation."""
    def __init__(self, method: str, lengthN: int, *, sdus_a: int | None = None,
                 lfsr_seed: int = 0xA5, A_override: np.ndarray | None = None):
        self.method = method.lower()
        self.lengthN = lengthN
        if A_override is not None:
            self.A = np.asarray(A_override, dtype=np.int32)
            return
        if self.method == 'adus':
            self.A = np.arange(lengthN, dtype=np.int32)
        elif self.method == 'sdus':
            if sdus_a is None:
                sdus_a = a_dict.get(lengthN)
                if sdus_a is None:
                    raise ValueError(f"No SDUS 'a' for lengthN={lengthN}")
            self.A = generate_sdus(lengthN, sdus_a)
        elif self.method == 'random8':
            self.A = random8_thresholds(lengthN)
        elif self.method == 'lfsr':
            self.A = generate_lfsr_sequence_auto(seed=lfsr_seed, lengthN=lengthN)
        elif self.method == 'halton':
            A1, _ = halton_thresholds_pair(lengthN)
            self.A = A1
        elif self.method == 'sobol':
            A1, _ = sobol_thresholds_pair(lengthN)
            self.A = A1
        else:
            raise ValueError(f"Unknown method {method}")

    def bitstream(self, p_prob: float) -> np.ndarray:
        N = self.lengthN
        val_8bit = int(round(p_prob * 255.0))  # 0..255
        if val_8bit <= 0:
            return np.zeros(N, dtype=np.uint8)     # p=0 饱和
        if val_8bit >= 255:
            return np.ones(N, dtype=np.uint8)      # p=1 饱和
        num = val_8bit * (N - 1)                  # 左值整数
        rhs = (self.A.astype(np.int64) * 255)     # 右值整数
        return (num >= rhs).astype(np.uint8)

# =====================
# Pair generator (pixel & weight) with fresh/fixed + uMUL support
# =====================
class BitstreamGeneratorPair:
    """
    A pair of generators for (pixel, weight).
    methods: 'adus_sdus','sobol','halton','lfsr','random8','umul','htc','vlsi22','tu','tub2'
    qmc_mode:
      - 'fixed' : Sobol/Halton/LFSR/Random8 用一对固定序列（整张图复用）
      - 'fresh' : Sobol/Halton/Random8 每次乘法都“2D 成对拉一批”，A1→pixel，A2→weight
      - DUS 始终固定
    """
    def __init__(self, method: str, lengthN: int, *, sdus_a: int | None = None,
                 lfsr_seed1: int = 0xA5, lfsr_seed2: int = 0xC3, qmc_mode: str = 'fixed'):
        self.method = method.lower()
        self.lengthN = lengthN
        self.sdus_a = sdus_a if sdus_a is not None else a_dict.get(lengthN)
        if self.method == 'adus_sdus' and self.sdus_a is None:
            raise ValueError(f"No SDUS 'a' for lengthN={lengthN}")
        if qmc_mode not in ('fixed', 'fresh'):
            raise ValueError("qmc_mode must be 'fixed' or 'fresh'")
        self.qmc_mode = qmc_mode
        self._is_fresh = (qmc_mode == 'fresh')

        # Members / engines / caches
        self.pix = None
        self.wgt = None
        self._eng_pix = None
        self._eng_wgt = None
        self._lfsr_pix = None
        self._lfsr_wgt = None
        # pairwise caches for fresh (store a tuple (A1, A2))
        self._pair_cache = {'sobol': None, 'halton': None, 'random8': None}
        self._eng2d_sobol = None
        self._eng2d_halton = None
        self._rng_pair = None

        # ---- method-specific init ----
        if self.method == 'adus_sdus':
            self.pix = BitstreamGenerator('adus',  lengthN)
            self.wgt = BitstreamGenerator('sdus',  lengthN, sdus_a=self.sdus_a)

        elif self.method == 'sobol':
            if not self._is_fresh:
                A1, A2 = sobol_thresholds_pair(lengthN)
                self.pix = BitstreamGenerator('sobol', lengthN, A_override=A1)
                self.wgt = BitstreamGenerator('sobol', lengthN, A_override=A2)
            else:
                self._eng2d_sobol = Sobol(d=2, scramble=False, seed=None)

        elif self.method == 'halton':
            if not self._is_fresh:
                A1, A2 = halton_thresholds_pair(lengthN)
                self.pix = BitstreamGenerator('halton', lengthN, A_override=A1)
                self.wgt = BitstreamGenerator('halton', lengthN, A_override=A2)
            else:
                self._eng2d_halton = Halton(d=2, scramble=False, seed=None)



     
        elif self.method == 'lfsr':
            if not self._is_fresh:
                self.pix = BitstreamGenerator('lfsr', lengthN, lfsr_seed=lfsr_seed1)
                self.wgt = BitstreamGenerator('lfsr', lengthN, lfsr_seed=lfsr_seed2)
            else:
                bit_width = max(4, int(np.ceil(np.log2(lengthN + 1))))
                taps_pix, taps_wgt = get_two_lfsr_taps(bit_width)  # 不同本原多项式
                mask = (1 << bit_width) - 1
        
                def _pick_decimation_step(w: int) -> int:
                    T = (1 << w) - 1
                    d = max(3, int(np.sqrt(T)) | 1)  # 奇数起步
                    while math.gcd(d, T) != 1 or d in (1, T-1):
                        d += 2
                    return d
        
                step_pix = 1
                step_wgt = 1 if taps_wgt != taps_pix else _pick_decimation_step(bit_width)
        
                # 保存状态（不同 seed + 不同 taps + 可能不同跳步）
                self._lfsr_pix = {'sr': (lfsr_seed1 & mask) or 1, 'taps': taps_pix, 'mask': mask, 'step': step_pix}
                self._lfsr_wgt = {'sr': (lfsr_seed2 & mask) or 1, 'taps': taps_wgt, 'mask': mask, 'step': step_wgt}
        



                # 16-bit LFSR 的全周期
                T = (1 << bit_width) - 1  # 65535
                
                def _pick_inc(seed: int) -> int:
                    # 从 seed 派生一个 1..T-1 的增量，并确保和 T 互素（避免短周期）
                    inc = (seed * 0x9E3779B1) % T or 1
                    while math.gcd(inc, T) != 1:
                        inc = (inc + 1) % T or 1
                    return inc
                
                self._phase_inc_pix = _pick_inc(lfsr_seed1)
                self._phase_inc_wgt = _pick_inc(lfsr_seed2)
                # 可选：也存下 T 以备调试
                self._period_T = T


        elif self.method == 'random8':
            if not self._is_fresh:
                A1, A2 = random8_thresholds_pair(lengthN)
                self.pix = BitstreamGenerator('random8', lengthN, A_override=A1)
                self.wgt = BitstreamGenerator('random8', lengthN, A_override=A2)
            else:
                self._rng_pair = np.random.default_rng()

        elif self.method == 'umul':
            if not self._is_fresh:
                A1, A2 = sobol_thresholds_pair(lengthN)
                B1, B2 = sobol_thresholds_pair(lengthN)
                self.pix = BitstreamGenerator('sobol', lengthN, A_override=A1)
                self._umul_A_one  = A2.astype(np.int64)
                self._umul_A_zero = B1.astype(np.int64)
                self._umul_idx_one  = 0
                self._umul_idx_zero = 0
            else:
                self._eng_pix  = Sobol(d=1, scramble=False, seed=None)
                self._eng_wgt1 = Sobol(d=1, scramble=False, seed=None)
                self._eng_wgt0 = Sobol(d=1, scramble=False, seed=None)

        elif self.method == 'htc':
            assert _is_power_of_two(lengthN), "HTC 需要 lengthN=2^n"
            self._htc_sel = htc_build_fsm_schedule_sobol(lengthN)

        elif self.method == 'vlsi22':
            pass

        elif self.method in ('tu', 'tub2'):
            pass

        else:
            raise ValueError(f"Unknown pair method {method}")




    @staticmethod
    def _lfsr_next_state(state: dict, N: int) -> np.ndarray:
        mask = state['mask']
        taps = state['taps']
        sr   = state['sr'] or 1
        step = state.get('step', 1)
    
        vals = np.empty(N, dtype=np.uint32)
        for i in range(N):
            # decimation: 每个样本跨 step 个 LFSR 拍
            for _ in range(step):
                fb = 0; s = sr
                for t in taps:
                    fb ^= (s >> (t - 1)) & 1
                sr = ((sr << 1) & mask) | fb
                if sr == 0: sr = 1
            vals[i] = sr
    
        state['sr'] = sr
        A = (vals * (N - 1) // mask).astype(np.int32)
        return A
    @staticmethod
    def _lfsr_advance_n(state: dict, k: int) -> None:
        mask = state['mask']; taps = state['taps']
        sr   = state['sr'] or 1
        step = state.get('step', 1)
        for _ in range(k):
            for __ in range(step):
                fb = 0; s = sr
                for t in taps:
                    fb ^= (s >> (t - 1)) & 1
                sr = ((sr << 1) & mask) | fb
                if sr == 0: sr = 1
        state['sr'] = sr


    # ---- draw a pair of thresholds for fresh (and cache them) ----
    def _draw_pair_thresholds(self, which: str):
        N = self.lengthN
        which = which.lower()
        if which == 'sobol':
            assert self._eng2d_sobol is not None

            S = self._eng2d_sobol.random(n=N)
            A1 = np.clip((S[:, 0] * N).astype(int), 0, N - 1).astype(np.int32)
            A2 = np.clip((S[:, 1] * N).astype(int), 0, N - 1).astype(np.int32)
            self._pair_cache['sobol'] = (A1, A2)
        elif which == 'halton':
            assert self._eng2d_halton is not None
            H = self._eng2d_halton.random(n=N)
            A1 = np.clip((H[:, 0] * N).astype(int), 0, N - 1).astype(np.int32)
            A2 = np.clip((H[:, 1] * N).astype(int), 0, N - 1).astype(np.int32)
            self._pair_cache['halton'] = (A1, A2)
        elif which == 'random8':
            assert self._rng_pair is not None
            R = self._rng_pair.integers(0, 256, size=(2, N), dtype=np.int32)
            A1 = (R[0] * (N - 1) // 255).astype(np.int32)
            A2 = (R[1] * (N - 1) // 255).astype(np.int32)
            self._pair_cache['random8'] = (A1, A2)
        else:
            raise ValueError(f"Unsupported pairwise draw: {which}")

    # ---- pixel() ----
    def pixel(self, p_prob: float) -> np.ndarray:
        N = self.lengthN

        def _bs_from_A(A: np.ndarray, method_name: str) -> np.ndarray:
            gen = BitstreamGenerator(method_name, N, A_override=A)
            return gen.bitstream(p_prob)

        m = self.method

        if m == 'adus_sdus':
            return self.pix.bitstream(p_prob)

        if m == 'sobol':
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache['sobol'] is None or self._pair_cache['sobol'][0] is None:
                self._draw_pair_thresholds('sobol')
            A1, A2 = self._pair_cache['sobol']
            self._pair_cache['sobol'] = (None, A2)
            return _bs_from_A(A1, 'sobol')

        if m == 'halton':
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache['halton'] is None or self._pair_cache['halton'][0] is None:
                self._draw_pair_thresholds('halton')
            A1, A2 = self._pair_cache['halton']
            self._pair_cache['halton'] = (None, A2)
            return _bs_from_A(A1, 'halton')


        # pixel()
        if m == 'lfsr':
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            # 先把像素通道的 LFSR 前进一个“相位增量”
            self._lfsr_advance_n(self._lfsr_pix, self._phase_inc_pix)
            # 然后从这个新相位开始，取连续 N 个状态，线性缩放到 0..N-1 做阈值
            A = self._lfsr_next_state(self._lfsr_pix, N)
            return _bs_from_A(A, 'lfsr')



        if m == 'random8':
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache['random8'] is None or self._pair_cache['random8'][0] is None:
                self._draw_pair_thresholds('random8')
            A1, A2 = self._pair_cache['random8']
            self._pair_cache['random8'] = (None, A2)
            return _bs_from_A(A1, 'random8')

        if m == 'umul':
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            S1 = self._eng_pix.random(n=N)[:, 0]
            A1 = np.clip((S1 * N).astype(int), 0, N - 1).astype(np.int32)
            return _bs_from_A(A1, 'sobol')

        if m == 'htc':
            return htc_encode_TB(p_prob, N)

        if m == 'vlsi22':
            return vlsi22_generate_stream_for_value(p_prob, N)

        raise ValueError(f"Unknown method {m}")

    # ---- weight() ----
    def weight(self, w_prob: float, pixel_stream: np.ndarray | None = None) -> np.ndarray:
        N = self.lengthN

        def _bs_from_A_with_prob(A: np.ndarray, method_name: str, prob: float) -> np.ndarray:
            gen = BitstreamGenerator(method_name, N, A_override=A)
            return gen.bitstream(prob)

        m = self.method

        if m in ('adus_sdus', 'sobol', 'halton', 'lfsr', 'random8', 'htc', 'vlsi22'):
            if m == 'adus_sdus':
                return self.wgt.bitstream(w_prob)

            if m == 'sobol':
                if not self._is_fresh:
                    return self.wgt.bitstream(w_prob)
                if self._pair_cache['sobol'] is None or self._pair_cache['sobol'][1] is None:
                    self._draw_pair_thresholds('sobol')
                A1, A2 = self._pair_cache['sobol']
                self._pair_cache['sobol'] = None
                return _bs_from_A_with_prob(A2, 'sobol', w_prob)

            if m == 'halton':
                if not self._is_fresh:
                    return self.wgt.bitstream(w_prob)
                if self._pair_cache['halton'] is None or self._pair_cache['halton'][1] is None:
                    self._draw_pair_thresholds('halton')
                A1, A2 = self._pair_cache['halton']
                self._pair_cache['halton'] = None
                return _bs_from_A_with_prob(A2, 'halton', w_prob)

           
            # weight()
            if m == 'lfsr':
                if not self._is_fresh:
                    return self.wgt.bitstream(w_prob)
                
                self._lfsr_advance_n(self._lfsr_wgt, self._phase_inc_wgt)
                A = self._lfsr_next_state(self._lfsr_wgt, N)
                return _bs_from_A_with_prob(A, 'lfsr', w_prob)




            if m == 'random8':
                if not self._is_fresh:
                    return self.wgt.bitstream(w_prob)
                if self._pair_cache['random8'] is None or self._pair_cache['random8'][1] is None:
                    self._draw_pair_thresholds('random8')
                A1, A2 = self._pair_cache['random8']
                self._pair_cache['random8'] = None
                return _bs_from_A_with_prob(A2, 'random8', w_prob)

            if m == 'htc':
                return htc_encode_RB_by_FSM(w_prob, N, self._htc_sel)

            if m == 'vlsi22':
                return vlsi22_generate_stream_for_value(w_prob, N)

        if m == 'umul':
            val8 = int(round(w_prob * 255.0))
            if val8 <= 0:
                return np.zeros(N, dtype=np.uint8)
            if val8 >= 255:
                return np.ones(N, dtype=np.uint8)

            num = val8 * (N - 1)
            rhs_scale = 255
            out = np.empty(N, dtype=np.uint8)
            if pixel_stream is None:
                pixel_stream = np.zeros(N, dtype=np.uint8)

            if not self._is_fresh:
                A1 = self._umul_A_one
                A0 = self._umul_A_zero
                i1 = self._umul_idx_one
                i0 = self._umul_idx_zero
                for i in range(N):
                    if pixel_stream[i] == 1:
                        out[i] = 1 if num >= A1[i1] * rhs_scale else 0
                        i1 = (i1 + 1) % N
                    else:
                        out[i] = 1 if num >= A0[i0] * rhs_scale else 0
                        i0 = (i0 + 1) % N
                self._umul_idx_one = i1
                self._umul_idx_zero = i0
                return out
            else:
                for i in range(N):
                    if pixel_stream[i] == 1:
                        t = int(self._eng_wgt1.random(n=1)[0, 0] * N)
                    else:
                        t = int(self._eng_wgt0.random(n=1)[0, 0] * N)
                    t = min(max(t, 0), N - 1)
                    out[i] = 1 if num >= t * rhs_scale else 0
                return out

        raise ValueError(f"Unknown method {m}")


# =====================
# Logic primitives
# =====================
def F_mul(a, b, mode: str = 'xnor'):
    if mode == 'xnor':
        return np.logical_not(np.logical_xor(a, b)).astype(np.uint8)
    elif mode == 'and':
        return np.logical_and(a, b).astype(np.uint8)
    elif mode == 'exact':
        return a * b
    else:
        raise ValueError(f"Unsupported multiplication mode: {mode}")

def F_add_mux(s1: np.ndarray, s2: np.ndarray, bipolar: bool = True) -> float:
    """按位二选一 MUX：每位随机 0/1，0取s1，1取s2；输出单极性均值，再可选转双极性（半和）"""
    s1 = np.asarray(s1, dtype=np.uint8)
    s2 = np.asarray(s2, dtype=np.uint8)
    assert s1.shape == s2.shape, "s1 与 s2 长度需一致"
    sel = np.random.randint(0, 2, size=len(s1), dtype=np.uint8)  # 0 or 1
    combined = s1 * (1 - sel) + s2 * sel
    mean_p = combined.mean()
    return (2.0 * mean_p - 1.0) if bipolar else float(mean_p)
def uSADD_stream(streams: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Unary Scaled ADD（流式逐拍实现）：
    - 每拍并行计数 N 输入的 '1' 个数 c_t
    - 累加器 acc += c_t；当 acc >= N 时输出1，并 acc -= N；否则输出0
    返回：与输入等长的 {0,1} 位流，且 1 的总数 = floor(ΣK / N)
    """
    assert len(streams) >= 1, "至少需要1个输入流"
    L = len(streams[0])
    for s in streams:
        assert len(s) == L, "所有输入位流长度必须一致"
    S = np.stack([s.astype(np.uint8) for s in streams], axis=0)   # (N, L)
    counts = np.sum(S, axis=0).astype(np.int32)                   # c_t
    out = np.zeros(L, dtype=np.uint8)
    acc = 0
    N = S.shape[0]
    for t in range(L):
        acc += int(counts[t])
        if acc >= N:
            out[t] = 1
            acc -= N
        # else 0
    return out

def uSADD_sum_value_from_streams(streams: list[np.ndarray] | tuple[np.ndarray, ...],
                                 bipolar: bool = True) -> float:
    """
    等价替代：不再生成逐拍 uSADD 位流，直接用总 '1' 个数做精确整除。
    - out_ones = floor(sum_i K_i / K)，其中 K_i 是第 i 条流的 '1' 个数，K 是流条数
    - out_mean = out_ones / L
    返回与原实现相同口径的“真和”（双极/单极）
    """
    assert len(streams) >= 1, "至少需要1个输入流"
    L = len(streams[0])
    for s in streams:
        assert len(s) == L, "所有输入位流长度必须一致"
    K = len(streams)
    # 总 '1' 个数
    ones_total = int(sum(int(np.sum(np.asarray(s, dtype=np.uint8))) for s in streams))
    # uSADD 输出流的 '1' 个数 = floor(ones_total / K)
    out_ones = ones_total // K
    p_mean = out_ones / float(L)  # 单极均值
    if bipolar:
        x_mean = 2.0 * p_mean - 1.0
        return K * x_mean
    else:
        return K * p_mean


# =====================
# Evaluation on image (exact + SC with pairs)
# =====================
def evaluate_methods_on_image(img_path: str,
                              methods=('adus_sdus','sobol','halton','lfsr','random8','umul','tu','tub2'),
                              lengthN: int = 256,
                              invert_output: bool = True,
                              save_dir: str = './SC_Outputs',
                              fresh_for=('sobol','halton','lfsr','random8','umul')):
    os.makedirs(save_dir, exist_ok=True)

    # SDUS parameter check
    sdus_a = a_dict.get(lengthN)
    if sdus_a is None and ('adus_sdus' in [m.lower() for m in methods]):
        raise ValueError(f"No SDUS 'a' for lengthN={lengthN}")

    # Load image
    # Load image: keep RGB for display, L for processing
    img_rgb = Image.open(img_path).convert('RGB')
    orig_color = np.array(img_rgb)                                     # for panel display
    gray = np.array(img_rgb.convert('L'), dtype=float) / 255.0         # for processing

    H, W = gray.shape

    # ===== 3×3 卷积核（Sobel）+ padding =====
    Kx, Ky, norm_factor = get_kernels_3x3('sobel')        # <<< 新核
    pad = np.pad(gray, ((1,1), (1,1)), mode='reflect')    # <<< 3×3 需要四周各 1

    # ---------- EXACT (bipolar) ----------
    exact_mag = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            gx = 0.0; gy = 0.0
            # Kx
            for m in range(3):
                for n in range(3):
                    w = Kx[m, n]
                    if w == 0.0: continue
                    p = float(pad[i + m, j + n])          # <<< 3×3 索引
                    p_bi = unipolar_to_bipolar(p)
                    gx += F_mul(p_bi, float(w), mode='exact')
            # Ky
            for m in range(3):
                for n in range(3):
                    w = Ky[m, n]
                    if w == 0.0: continue
                    p = float(pad[i + m, j + n])
                    p_bi = unipolar_to_bipolar(p)
                    gy += F_mul(p_bi, float(w), mode='exact')

            exact_mag[i, j] = math.sqrt(gx*gx + gy*gy)

    # 把 |G| 归一到 0..255：除以理论上界 norm_factor 再 ×255
    exact_mag = (exact_mag / max(1e-12, norm_factor)) * 255.0   # <<< 替代原来的 2*sqrt(2)
    exact_mag = np.clip(exact_mag, 0.0, 255.0)
    exact_disp = 255.0 - exact_mag if invert_output else exact_mag
    exact_disp = exact_disp.astype(np.uint8)

    exact_path = os.path.join(save_dir, f"conv3x3_EXACT_N{lengthN}.png")  # <<< 文件名可改
    Image.fromarray(exact_disp).save(exact_path)
    print(f"Saved EXACT baseline: {exact_path}")


    # ---------- SC (per method pair) ----------
    def method_label(m: str) -> str:
        mapping = {
            'adus_sdus':'DUS (proposed)',
            'sobol':'Sobol',
            'halton':'Halton1_Halton2',
            'lfsr':'LFSR',
            'random8':'Random8_1_Random8_2',
            'umul':'uGEMM‘2020',
            'htc':'HTC‘2025',
            'vlsi22':'Downscale‘2022',
            'tu':'TU_TemporalUnary',
            'tub2':'TubGEMM‘2023',
        }
        return mapping.get(m.lower(), m.upper())

    expand_times = max(1, lengthN // 256)
    results: dict[str, dict] = {}
    
    for method in methods:
        mlow = method.lower()
    
        # ===== TU/TUB2：计数域专用路径 =====
        if mlow in ('tu', 'tub2'):
            tu_mag = conv3x3_temporal_unary_counts(gray, Kx, Ky, lengthN)  # 未归一幅值
            sc_mag = (tu_mag / max(1e-12, norm_factor)) * 255.0
            sc_mag = np.clip(sc_mag, 0.0, 255.0)
            sc_disp = (255.0 - sc_mag) if invert_output else sc_mag
            sc_disp = sc_disp.astype(np.uint8)
    
            mae = float(np.mean(np.abs(sc_disp.astype(float)/255.0 - exact_disp.astype(float)/255.0)))
            label = method_label(method)
    
            maxk = int(np.round(gray * lengthN).max())
            latency_note = f"{maxk}" if mlow == 'tu' else f"{(maxk + 1)//2}"
    
            results[label] = {'mae': mae, 'image': sc_disp, 'latency': latency_note}
            out_path = os.path.join(save_dir, f"conv3x3_{label}_N{lengthN}_MAE{mae:.4f}_LAT{latency_note}.png")
            Image.fromarray(sc_disp).save(out_path)
            print(f"Saved: {out_path}  |  MAE={mae:.6e} | LAT={latency_note}")
            continue
    
        # ===== 其它方法：位流/XNOR/uSADD 路径 =====
        qmc_mode = 'fresh' if mlow in fresh_for else 'fixed'
        pair = BitstreamGeneratorPair(method, lengthN, sdus_a=sdus_a, qmc_mode=qmc_mode)
        sc_mag = np.zeros((H, W), dtype=float)
    
        # VLSI’22 的长度检查：P 必须是完全平方
        if mlow == 'vlsi22':
            q = int(np.sqrt(lengthN))
            assert q * q == lengthN, "VLSI’22 需要 lengthN 为完全平方（如 256、1024）"
    
        for i in range(H):
            for j in range(W):
                gx = 0.0; gy = 0.0
                for (mat, assign) in ((Kx,'gx'), (Ky,'gy')):
                    streams = []
                    for m2 in range(3):
                        for n2 in range(3):
                            w = mat[m2, n2]
                            if w == 0.0: 
                                continue
                            p = float(pad[i + m2, j + n2])
                            p_prob = p
                            w_prob = bipolar_to_unipolar(float(w))  # [-1,1] → [0,1]
    
                            # >>> 这里是关键改动：vlsi22 单独分支 <<<
                            if mlow == 'vlsi22':
                                # 成对生成 + 乘法级误差补偿，不再做 replicate/rotate
                                s_pix, s_wgt = vlsi22_pair_streams(p_prob, w_prob, lengthN)
                                s_pix_e = s_pix
                                s_wgt_e = s_wgt
                            else:
                                # 其余方法保持原策略（像素 replicate，权值 rotate 去相关）
                                s_pix = pair.pixel(p_prob)
                                s_pix_e = replicate_expand(s_pix, expand_times)
    
                                if mlow == 'umul':
                                    s_wgt = pair.weight(w_prob, pixel_stream=s_pix)
                                else:
                                    s_wgt = pair.weight(w_prob)
                                s_wgt_e = rotate_expand(s_wgt, expand_times)
    
                            streams.append(F_mul(s_pix_e, s_wgt_e, mode='xnor'))
    
                    # N 路 uSADD（双极） → 当前卷积核方向的“真和”
                    val = uSADD_sum_value_from_streams(streams, bipolar=True)
                    if assign == 'gx': 
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
        results[label] = {'mae': mae, 'image': sc_disp}
    
        out_path = os.path.join(save_dir, f"conv3x3_{label}_N{lengthN}_MAE{mae:.4f}.png")
        Image.fromarray(sc_disp).save(out_path)
        print(f"Saved: {out_path}  |  MAE={mae:.6e}")
    
    
    
    # === 3×3 single-column panel (IEEE) ===
    # 首选展示顺序（如缺项会自动跳过；多余的会填到最后）
    prefer_order = [
        'DUS (proposed)',   # ADUS_SDUS
        'LFSR' ,             # 第7格常用备选；也可换成 Random8 或 Halton
        'Sobol',
        'uGEMM‘2020',       # 原 uMUL
        'TubGEMM‘2023',     # TUB2
        'Downscale‘2022',   # VLSI'22
        'HTC‘2025',         # ASPDAC'25
       
    ]
    # 用 prefer_order 组装最多 7 个方法（前两格是 Original/Exact）
    ordered = [k for k in prefer_order if k in results][:7]
    # 若还没凑满 7 个，就从剩余结果里补齐
    if len(ordered) < 7:
        extras = [k for k in results.keys() if k not in ordered]
        ordered += extras[:(7 - len(ordered))]
    
    # 建图：单栏宽度 ~3.5 in；高度略增以容纳标题
    fig, axes = plt.subplots(3, 3, figsize=(3.5, 3.9), dpi=600)
    axes = axes.ravel()
    
    # (1) 原图（彩色）
    axes[0].imshow(orig_color)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # (2) Exact（灰度）
    axes[1].imshow(exact_disp, cmap='gray')
    axes[1].set_title('Exact (FP)', fontsize=12)
    axes[1].axis('off')
    
    # (3..9) 方法结果（灰度）
    for idx, name in enumerate(ordered, start=2):
        ax = axes[idx]
        ax.imshow(results[name]['image'], cmap='gray')
        ax.set_title(f"{name}\nMAE={results[name]['mae']:.4f}", fontsize=12)
        ax.axis('off')
    
    # 隐藏未用到的格子
    for i in range(2 + len(ordered), 9):
        axes[i].axis('off')
    
    # 紧凑排版：减小子图间距，适配单栏
    plt.subplots_adjust(wspace=0.05, hspace=0.18)
    
    panel_path = os.path.join(save_dir, f"panel3x3_N{lengthN}.png")
    plt.savefig(panel_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved panel: {panel_path}")

    
    
    print("\n=== MAE Summary (vs EXACT) ===")
    for name, info in results.items():
        lat = f", LAT={info['latency']}" if 'latency' in info else ""
        print(f"{name:>22}: MAE = {info['mae']:.6e}{lat}")
    
    return exact_disp, results


# =====================
# Main
# =====================
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _env_summary() -> dict:
    info = {
        "USE_GPU": int(USE_GPU),
        "BSDS_ROOT": os.environ.get("BSDS_ROOT", ""),
        "python": os.sys.version.replace("\n", " "),
        "numpy": np.__version__,
    }
    # Optional: CuPy
    try:
        import cupy as cp  # type: ignore
        info["cupy"] = cp.__version__
        try:
            info["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
            info["gpu_count"] = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            pass
    except Exception:
        info["cupy"] = None
    return info


def parse_args():
    ap = argparse.ArgumentParser(description="BSDS500 Canny evaluation (SC / FP baselines, CUDA optional)")
    ap.add_argument("--bsds_root", default=os.environ.get("BSDS_ROOT", "./datasets/BSR"),
                    help="Root folder that contains BSDS500/ (e.g., /data/xiashiyu/datasets/BSR/BSR).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="BSDS500 split.")
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", "./SC_Outputs"),
                    help="Output directory for PR curves and CSV.")
    ap.add_argument("--tag", default="", help="Optional run tag (default: timestamp).")

   
    # - fresh-mode Sobol/Halton uses seed=None by design -> still changes each draw.
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="Global numpy/random seed (fixed-mode reproducibility).")

    # Sweep
    ap.add_argument("--high_start", type=float, default=10.0)
    ap.add_argument("--high_stop",  type=float, default=180.0)
    ap.add_argument("--high_num",   type=int,   default=64)
    ap.add_argument("--low_ratio",  type=float, default=0.4)
    ap.add_argument("--tol_frac",   type=float, default=0.0075)
    ap.add_argument("--gauss_sigma", type=float, default=1.0)
    ap.add_argument("--max_images", type=int, default=-1, help="-1 means all images.")

    # Runtime
    ap.add_argument("--use_gpu", type=int, choices=[0, 1], default=None,
                    help="Override USE_GPU (0: CPU, 1: GPU). If omitted, uses env USE_GPU (default=1).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Optional runtime override (only affects runtime branches that reference USE_GPU)
    # NOTE: import branches already happened. So this is mainly for consistent logging and

    if args.use_gpu is not None:
        os.environ["USE_GPU"] = str(int(args.use_gpu))
        # keep local variable aligned
        globals()["USE_GPU"] = bool(args.use_gpu)

    # Seed (keeps fixed-mode reproducible; fresh-mode intentionally varies)
    np.random.seed(args.seed)
    random.seed(args.seed)

    bsds_root = args.bsds_root
    split = args.split

    tag = args.tag.strip() or _now_tag()
    out_dir = os.path.join(args.out_dir, f"run_{tag}")
    _ensure_dir(out_dir)

    # Record metadata for paper artifact traceability
    meta = {
        "tag": tag,
        "bsds_root": bsds_root,
        "split": split,
        "seed": args.seed,
        "use_gpu": int(USE_GPU),
        "sweep": {
            "high_start": args.high_start,
            "high_stop": args.high_stop,
            "high_num": args.high_num,
            "low_ratio": args.low_ratio,
            "tol_frac": args.tol_frac,
            "gauss_sigma": args.gauss_sigma,
            "max_images": None if args.max_images < 0 else args.max_images,
        },
        "env": _env_summary(),
        "notes": {
            "fresh_mode": "For sobol/halton/lfsr/random8/umul inside sc_sobel_gxgy, qmc_mode='fresh' remains stochastic by design.",
        },
    }
    _write_json(os.path.join(out_dir, "run_meta.json"), meta)


    methods = [
        ("fp", "exact",    256),   # FP baseline
        ("sc", "adus_sdus", 256),  # DUS
        ("sc", "sobol",    256),
        ("sc", "umul",     256),
        ("sc", "tub2",     256),
        ("sc", "vlsi22",   256),
        ("sc", "htc",      256),
    ]

    high_grid = np.linspace(args.high_start, args.high_stop, args.high_num)

    for mode, method, N in methods:
        tag2 = f"{mode}_{method}_N{N}"
        out_png = os.path.join(out_dir, f"PR_{tag2}.png")
        out_csv = os.path.join(out_dir, f"PR_{tag2}.csv")

        res = evaluate_bsds_canny(
            bsds_root, split, mode, method=method, lengthN=N,
            gauss_sigma=args.gauss_sigma,
            high_grid=high_grid,
            low_ratio=args.low_ratio,
            tol_frac=args.tol_frac,
            max_images=None if args.max_images < 0 else args.max_images,
            save_plot=out_png,
            save_csv=out_csv
        )

        ODS_P, ODS_R, ODS_F, ODS_th = res["ODS"]
        OIS_meanF, OIS_stdF = res["OIS"]
        AP = res["AP"]
        print(f"[{tag2}]  ODS(F)={ODS_F:.3f} @th={ODS_th:.1f} | OIS(F)={OIS_meanF:.3f} (±{OIS_stdF:.3f}) | AP={AP:.3f}")
        print(f"         PR plot: {out_png}")
        print(f"         PR data: {out_csv}")

    print(f"\nDone. All outputs are under: {out_dir}")
