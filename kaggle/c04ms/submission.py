"""Kaggle submission: multi-resolution self-ensemble of c04 (YOLO26s, trained at 1920): the same
weights exported at 1920, 2112 and 2304 input width (1.0x, 1.1x, 1.2x training size), fused with
WBF (IoU 0.4, conf = sum / 3). Cutoff 0.28 = c04's own val-tuned cutoff; not tuned for this
ensemble (no local val run, by choice).

Pushed with `kaggle kernels push -p kaggle/c04ms`; otherwise identical to kaggle/ens4/submission.py.
"""

import gc
import glob
import os
import subprocess
import sys
import time

# Placements to try in order: (models, GPU per model, fused-conf cutoff); the second is a fallback
# if the first fails its warm-up (all three inputs are small, so either should fit).
PLACEMENTS = [
    (["c04_1920", "c04_2112", "c04_2304"], [0, 1, 1], 0.28),
    (["c04_1920", "c04_2112", "c04_2304"], [0, 0, 1], 0.28),
]
WBF_THR = 0.4

D = os.path.dirname(glob.glob("/kaggle/input/**/c04_1920.onnx", recursive=True)[0])
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
            ens = Ensemble([D + "/" + m + ".onnx" for m in names], devices, thr=WBF_THR)
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
