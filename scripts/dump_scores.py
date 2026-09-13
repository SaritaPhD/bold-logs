"""Train each detector once per condition (seed 0) and persist calibration/test
scores + labels + unit order, so thresholds (alpha sweep, oracle recalibration,
histograms) can be studied without retraining. Also an MLP-AE variant with the
OOV column removed (ablation for RQ2)."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, joblib
from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, IF, OCSVM, MLPAE, MarkovNext
from logad.eval import split
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "scores"); os.makedirs(OUT, exist_ok=True)
FILES = {"hdfs": "hdfs_sessions.joblib", "bgl_global": "bgl_global.joblib", "thunderbird": "thunderbird_global.joblib"}
cfgs = sys.argv[1:] or list(FILES)
for cfg in cfgs:
    df = joblib.load(os.path.join(DATA, FILES[cfg]))["df"]
    for mode in ["chrono", "random"]:
        train, cal, test = split(df, mode=mode, seed=0)
        fx = CountVectors().fit(train.seq.tolist())
        F = lambda d: np.hstack([fx.transform(d.seq.tolist()), seq_stats(d.seq.tolist())])
        Xtr, Xca, Xte = F(train), F(cal), F(test)
        V = len(fx.vocab)  # OOV column index
        y = test.label.to_numpy()
        store = {"y": y, "order_test": test.order.to_numpy(), "order_cal": cal.order.to_numpy(),
                 "oov_share_test": Xte[:, V] > 0}
        def run(name, mdl, Xt=Xtr, Xc=Xca, Xe=Xte):
            t0 = time.time(); mdl.fit(Xt)
            store[f"{name}_cal"] = mdl.score(Xc); store[f"{name}_test"] = mdl.score(Xe)
            print(f"{cfg}/{mode} {name} {time.time()-t0:.0f}s", flush=True)
        run("PCA", PCASPE()); run("AE", MLPAE(seed=0, max_iter=60)); run("IF", IF(seed=0))
        rng = np.random.default_rng(0); idx = rng.choice(len(Xtr), size=min(10000, len(Xtr)), replace=False)
        m = OCSVM().fit(Xtr[idx]); store["OCSVM_cal"] = m.score(Xca); store["OCSVM_test"] = m.score(Xte)
        mk = MarkovNext(k=2).fit_seqs(train.seq.tolist())
        store["Markov_cal"] = mk.score_seqs(cal.seq.tolist()); store["Markov_test"] = mk.score_seqs(test.seq.tolist())
        # ablation: AE without the OOV column
        keep = [i for i in range(Xtr.shape[1]) if i != V]
        run("AEnoOOV", MLPAE(seed=0, max_iter=60), Xtr[:, keep], Xca[:, keep], Xte[:, keep])
        np.savez_compressed(os.path.join(OUT, f"{cfg}_{mode}.npz"), **store)
print("done")
