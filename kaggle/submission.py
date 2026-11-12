"""Kaggle submission script (runs on the pinned 2021 image: Python 3.7, torch 1.9, T4).

Pushed with `kaggle kernels push -p <folder>` next to kernel-metadata.json. The attached dataset
holds the ONNX model, kaggle/cots_onnx.py, and an onnxruntime-gpu 1.11 wheel (internet is off).
Writes debug.csv (per-frame box counts and timing) next to submission.csv.
"""

import glob
import os
import subprocess
import sys
import time

MODEL = "c04_best.onnx"
CONF_THR = 0.28  # best full-val F2 cutoff for c04 (model_dev/c04_.../seed0/eval/best_f2.json)

D = os.path.dirname(glob.glob("/kaggle/input/**/" + MODEL, recursive=True)[0])
subprocess.run("pip install --no-index --no-deps {0}/onnxruntime_gpu-*.whl {0}/flatbuffers-*.whl".format(D), shell=True, check=True)
sys.path.insert(0, D)
sys.path.insert(0, glob.glob("/kaggle/input/**/greatbarrierreef", recursive=True)[0].rsplit("/", 1)[0])

import greatbarrierreef  # noqa: E402
from cots_onnx import OnnxDetector, to_annotation_string  # noqa: E402

det = OnnxDetector(D + "/" + MODEL, conf=0.01)
print("providers", det.sess.get_providers(), "input", det.in_h, det.in_w, flush=True)
assert det.sess.get_providers()[0] == "CUDAExecutionProvider", "GPU provider did not load"

env = greatbarrierreef.make_env()
log = open("debug.csv", "w")
log.write("n,shape,n_dets,n_submitted,annotations,all_dets,seconds\n")
t0 = time.time()
for n, (pixel_array, df) in enumerate(env.iter_test()):
    t = time.time()
    dets = det(pixel_array)  # RGB uint8, as the API delivers
    ann = to_annotation_string(dets, CONF_THR)
    df["annotations"] = ann
    env.predict(df)
    if n < 20 or n % 500 == 0:
        log.write('{},{},{},{},"{}","{}",{:.3f}\n'.format(n, "x".join(map(str, pixel_array.shape)), len(dets), len(ann.split()) // 5, ann,
                                                        to_annotation_string(dets, 0.0), time.time() - t))
log.close()
print("frames", n + 1, "total %.1f s" % (time.time() - t0), flush=True)
