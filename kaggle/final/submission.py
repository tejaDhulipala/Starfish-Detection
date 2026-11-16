"""Kaggle submission: final ensemble (8 votes, 10 passes), FP16 dynamic-shape ONNX, one session per model.

Votes (WBF, conf = sum / 8): c08 at 4 rotations (0/90/180/270 deg at long side 3648/3840/4032/3648),
c06 at 3648, c07 at 3840, c13 at 4032, and c17 at 1920 + 2112 + 2304 pre-fused into one vote.
Val (pre-retrain weights, passes at native size): F2 0.7583 at cutoff 0.16, held-out 0.7341.
The +200-400 px sizes for c08 / c07 / c13 follow 2nd place and were not validated.

c08 runs on GPU 0, everything else on GPU 1, in parallel. Prints a per-pass benchmark and GPU
memory on the first frame; writes debug.csv.
"""

import glob
import os
import subprocess
import sys
import time

CONF_THR = 0.16
WBF_THR = 0.4
PASSES = [  # (vote group, model, quarter turns CCW, long side)
    ("c08_0", "c08", 0, 3648), ("c08_90", "c08", 1, 3840), ("c08_180", "c08", 2, 4032), ("c08_270", "c08", 3, 3648),
    ("c06", "c06", 0, 3648), ("c07", "c07", 0, 3840), ("c13", "c13", 0, 4032),
    ("c17", "c17", 0, 1920), ("c17", "c17", 0, 2112), ("c17", "c17", 0, 2304),
]
GPU = {"c08": 0, "c06": 1, "c07": 1, "c13": 1, "c17": 1}

D = os.path.dirname(glob.glob("/kaggle/input/**/final_ensemble.py", recursive=True)[0])
subprocess.run("pip install --no-index --no-deps {0}/onnxruntime_gpu-*.whl {0}/flatbuffers-*.whl".format(D), shell=True, check=True)
sys.path.insert(0, D)
sys.path.insert(0, glob.glob("/kaggle/input/**/greatbarrierreef", recursive=True)[0].rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import greatbarrierreef  # noqa: E402
from cots_onnx import to_annotation_string  # noqa: E402
from final_ensemble import FinalEnsemble  # noqa: E402

n_gpu = torch.cuda.device_count()
print("gpus", n_gpu, flush=True)
gpu = {m: (g if n_gpu > 1 else 0) for m, g in GPU.items()}


def benchmark(ens, frame, reps=5):
    bgr = np.ascontiguousarray(frame[..., ::-1])
    for i, p in enumerate(ens.passes):
        t = time.time()
        for _ in range(reps):
            ens.run_pass(bgr, i)
        print("bench", p, "%.3f s/frame" % ((time.time() - t) / reps), flush=True)
    t = time.time()
    for _ in range(reps):
        ens(frame)
    print("bench ensemble end-to-end %.3f s/frame" % ((time.time() - t) / reps), flush=True)
    subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv", shell=True)


env = greatbarrierreef.make_env()
log = open("debug.csv", "w")
log.write("n,shape,n_dets,n_submitted,annotations,all_dets,seconds\n")
t0 = time.time()
for n, (pixel_array, df) in enumerate(env.iter_test()):
    if n == 0:
        ens = FinalEnsemble({m: D + "/" + m + ".onnx" for m in GPU}, gpu, PASSES, thr=WBF_THR)
        for m, d in ens.dets.items():
            assert d.sess.get_providers()[0] == "CUDAExecutionProvider", "GPU provider did not load for " + m
        ens(pixel_array)  # warm-up: cuDNN algorithm search per input shape, arena growth to peak
        ens(pixel_array)
        benchmark(ens, pixel_array)
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
