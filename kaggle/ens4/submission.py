"""Kaggle submission: WBF ensemble of c04 + c06 + c07 + c08 (val F2 0.7445 at fused cutoff 0.21).

Pushed with `kaggle kernels push -p kaggle/ens4`; runs on the pinned 2021 image (Python 3.7, 2x T4).
The attached dataset holds the four ONNX models, kaggle/cots_onnx.py, kaggle/cots_ensemble.py and
the onnxruntime-gpu 1.11 wheel. Models on different GPUs run in parallel; see PLACEMENTS for which
model goes where. Writes debug.csv (per-frame box counts and timing) and prints a per-model timing
benchmark and GPU memory on the first frame, to estimate the hidden-test runtime.
"""

import gc
import glob
import os
import subprocess
import sys
import time

# Placements to try in order: (models, GPU per model, fused-conf cutoff). Each 3648-input session
# keeps its own memory arena, and three of them on one T4 ran out of memory (kernel v1), so if a
# placement fails its warm-up, its sessions are freed and the next one is tried. The last one drops
# c07: c04+c06+c08 has val F2 0.7399 at cutoff 0.21 (vs 0.7445 for all four), and c04 + c06 are
# known to fit on one T4 together.
PLACEMENTS = [
    (["c04", "c06", "c07", "c08"], [0, 1, 1, 0], 0.21),
    (["c04", "c06", "c07", "c08"], [1, 1, 0, 0], 0.21),
    (["c04", "c06", "c08"], [1, 1, 0], 0.21),
]
WBF_THR = 0.4

D = os.path.dirname(glob.glob("/kaggle/input/**/c04_best.onnx", recursive=True)[0])
subprocess.run("pip install --no-index --no-deps {0}/onnxruntime_gpu-*.whl {0}/flatbuffers-*.whl".format(D), shell=True, check=True)
sys.path.insert(0, D)
sys.path.insert(0, glob.glob("/kaggle/input/**/greatbarrierreef", recursive=True)[0].rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import greatbarrierreef  # noqa: E402
from cots_ensemble import Ensemble  # noqa: E402
from cots_onnx import to_annotation_string  # noqa: E402

n_gpu = torch.cuda.device_count()
print("gpus", n_gpu, flush=True)


def benchmark(ens, names, frame, reps=10):
    bgr = np.ascontiguousarray(frame[..., ::-1])
    for m, d in zip(names, ens.dets):
        x, meta = d.preprocess(bgr)
        t = time.time()
        for _ in range(reps):
            d.run(x, meta, bgr.shape[:2])
        print("bench", m, "%.3f s/frame" % ((time.time() - t) / reps), flush=True)
    t = time.time()
    for _ in range(reps):
        ens(frame)
    print("bench ensemble end-to-end %.3f s/frame" % ((time.time() - t) / reps), flush=True)
    subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv", shell=True)


def build(frame):
    """First placement whose sessions load on the GPU and survive a warm-up and benchmark on frame."""
    for names, devices, thr in PLACEMENTS:
        if n_gpu < 2:
            devices = [0] * len(names)
        print("trying", names, devices, flush=True)
        ens = None
        try:
            ens = Ensemble([D + "/" + m + "_best.onnx" for m in names], devices, thr=WBF_THR)
            for m, d in zip(names, ens.dets):
                assert d.sess.get_providers()[0] == "CUDAExecutionProvider", "GPU provider did not load for " + m
            ens(frame)  # warm-up: cuDNN algorithm search and arena growth to peak
            ens(frame)
            benchmark(ens, names, frame)
            print("using", names, devices, "cutoff", thr, flush=True)
            return ens, thr
        except Exception as e:  # noqa: BLE001  (onnxruntime raises RuntimeException on OOM)
            print("placement failed:", repr(e)[:300], flush=True)
        if ens is not None:  # failed: free its sessions (and their GPU arenas) before the next try
            ens.pool.shutdown()
        del ens
        gc.collect()
        subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv", shell=True)
    raise RuntimeError("no placement fits")


env = greatbarrierreef.make_env()
log = open("debug.csv", "w")
log.write("n,shape,n_dets,n_submitted,annotations,all_dets,seconds\n")
t0 = time.time()
for n, (pixel_array, df) in enumerate(env.iter_test()):
    if n == 0:
        ens, CONF_THR = build(pixel_array)
    t = time.time()
    dets = ens(pixel_array)  # RGB uint8, as the API delivers
    ann = to_annotation_string(dets, CONF_THR)
    df["annotations"] = ann
    env.predict(df)
    if n < 20 or n % 500 == 0:
        log.write('{},{},{},{},"{}","{}",{:.3f}\n'.format(n, "x".join(map(str, pixel_array.shape)), len(dets), len(ann.split()) // 5, ann,
                                                        to_annotation_string(dets, 0.0), time.time() - t))
        log.flush()
log.close()
print("frames", n + 1, "total %.1f s" % (time.time() - t0), flush=True)
