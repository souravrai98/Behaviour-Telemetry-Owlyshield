"""Build the data/ folder of this package from the OwlyShield data rebuilt by
owly_reproduce_data.sh (OWLYSHIELD_CRAFT_REPRODUCTION_v1, labelled CSV
combined_labelled_25_model_features_20260929.csv, SHA-256 4a20aedf...ce03).

Not needed on the pod: data/ is shipped. Kept so the data can be rebuilt and checked.

  python3 prepare_owly_forecast.py --processed /path/to/calf-code-final-main/processed --out data

Scaling: the processed folders are standardised with CRAFT's scaler (fitted on the raw
benign training rows). That scaling is undone and a StandardScaler is refitted on all rows
of this package (training + benign test incl. validation + malicious test), as requested
for the forecasting runs (refit_on_all_rows). The arrays in data/ are written in that scale.

Sources:
  * training  : full benign training recordings (owlyshield_labelled_v4b_seq5_50pct/train*),
                i.e. the 808,637 rows BEFORE CRAFT's 1-in-16 cut. The cut is re-applied to
                window start positions at run time (owly_windows.thinned_starts).
  * benign    : the 56 whole benign test recordings (hard negatives and note-writing controls,
                owlyshield_labelled_v4b_lookalike/test*). CRAFT's medium test used the first
                81 rows of 483 pieces of these same 56 recordings; 81 rows cannot hold a
                96-row input, so the recordings are used whole here.
  * malicious : exactly the 120 attack recordings of CRAFT's medium test set
                (95 high-activity + 25 low-activity, each capped at its first 4,200 rows),
                taken from owlyshield_labelled_v4b_seq5_medium/test*.
  * validation: 10% of the benign test recordings, drawn per semantic label (hard negatives /
                note-writing controls) with seed 2021; the remaining 90% are the benign test.
                The large database-update validation set of CRAFT is not used.
"""
import argparse, hashlib, json, os, shutil
import numpy as np
import pandas as pd
from owly_windows import INPUT_LENGTH, HORIZONS, all_starts, thinned_starts

VAL_FRACTION = 0.10
VAL_SEED = 2021


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refit_on_all_rows(out):
    """Undo CRAFT's scaling (float64) and refit mean/std on training + benign + malicious
    rows; a zero std is set to 1, as sklearn's StandardScaler does."""
    sc = json.load(open(os.path.join(out, "scaler.json")))
    mu, sd = np.array(sc["mean"]), np.array(sc["scale"])
    names = ["train_rows", "benign_rows", "malicious_rows"]
    raw = {n: np.load(os.path.join(out, f"{n}.npy")).astype(np.float64) * sd + mu for n in names}
    allraw = np.concatenate([raw[n] for n in names])
    m2, v2 = allraw.mean(0), allraw.var(0)
    s2 = np.sqrt(v2); s2[s2 == 0] = 1.0
    for n in names:
        np.save(os.path.join(out, f"{n}.npy"), ((raw[n] - m2) / s2).astype(np.float32))
    sc.update(fit_scope="train + test rows of this package: 808,637 training + 185,995 benign test (incl. validation) + 36,848 malicious",
              mean=m2.tolist(), scale=s2.tolist(), var=v2.tolist(), n_samples_seen=int(len(allraw)))
    json.dump(sc, open(os.path.join(out, "scaler.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    P = args.processed
    LA = os.path.join(P, "owlyshield_labelled_v4b_lookalike")
    S50 = os.path.join(P, "owlyshield_labelled_v4b_seq5_50pct")
    MED = os.path.join(P, "owlyshield_labelled_v4b_seq5_medium")
    os.makedirs(args.out, exist_ok=True)

    # ---------------- training: full benign training recordings
    tr = np.load(os.path.join(S50, "train.npy")).astype(np.float32)
    tr_seg = np.load(os.path.join(S50, "train_segments.npy")).astype(np.int64)
    assert tr.shape == (808637, 24) and len(tr_seg) == 493
    assert not np.load(os.path.join(S50, "train_labels.npy")).any()
    # CRAFT's training rows are a subset of these rows (sanity check)
    ids50 = np.load(os.path.join(S50, "train_row_ids.npy"))
    idsm = np.load(os.path.join(MED, "train_row_ids.npy"))
    pos = {int(v): i for i, v in enumerate(ids50)}
    assert np.array_equal(tr[[pos[int(v)] for v in idsm]], np.load(os.path.join(MED, "train.npy")))

    # ---------------- benign test recordings (whole)
    la_test = np.load(os.path.join(LA, "test.npy")).astype(np.float32)
    la_lab = np.load(os.path.join(LA, "labels.npy")).ravel()
    la_seg = np.load(os.path.join(LA, "test_segments.npy")).astype(np.int64)
    la_ids = np.load(os.path.join(LA, "test_row_ids.npy"))
    segcsv = pd.read_csv(os.path.join(LA, "segments.csv"))
    tcsv = segcsv[segcsv.partition == "test"].set_index("output_start")
    ben_parts, ben_rows, start = [], [], 0
    for a, b in la_seg:
        if la_lab[a] != 0:
            continue
        assert (la_lab[a:b] == 0).all()
        info = tcsv.loc[int(a)]
        assert int(info.output_stop) == int(b) and int(info.binary_label) == 0
        ben_rows.append(la_test[a:b])
        ben_parts.append(dict(start=start, stop=start + int(b - a), rows=int(b - a),
                              source_file=info.source_file, semantic_label=info.semantic_labels))
        start += int(b - a)
    ben = np.concatenate(ben_rows)
    ben_info = pd.DataFrame(ben_parts)
    assert len(ben_info) == 56 and len(ben) == 185995

    # validation = 10% of benign test recordings, per semantic label, seed 2021
    rng = np.random.default_rng(VAL_SEED)
    role = np.array(["test"] * len(ben_info), dtype=object)
    for lab, grp in ben_info.groupby("semantic_label"):
        k = int(round(VAL_FRACTION * len(grp)))
        chosen = rng.choice(grp.index.to_numpy(), size=k, replace=False)
        role[np.sort(chosen)] = "validation"
    ben_info["role"] = role

    # ---------------- malicious: CRAFT medium test attacks
    m_test = np.load(os.path.join(MED, "test.npy")).astype(np.float32)
    m_lab = np.load(os.path.join(MED, "labels.npy")).ravel()
    m_seg = np.load(os.path.join(MED, "test_segments.npy")).astype(np.int64)
    m_ids = np.load(os.path.join(MED, "test_row_ids.npy"))
    la_first = {int(la_ids[a]): (a, b) for a, b in la_seg}
    mal_parts, mal_rows, start = [], [], 0
    for a, b in m_seg:
        if m_lab[a] != 1:
            continue
        assert (m_lab[a:b] == 1).all() and np.all(np.diff(m_ids[a:b]) == 1)
        src = la_first[int(m_ids[a])]                     # attack starts at its recording's first row
        info = tcsv.loc[int(src[0])]
        assert int(info.binary_label) == 1 and (b - a) == min(src[1] - src[0], 4200)
        assert np.array_equal(m_test[a:b], la_test[src[0]:src[0] + (b - a)])
        mal_rows.append(m_test[a:b])
        mal_parts.append(dict(start=start, stop=start + int(b - a), rows=int(b - a),
                              source_file=info.source_file, semantic_label=info.semantic_labels))
        start += int(b - a)
    mal = np.concatenate(mal_rows)
    mal_info = pd.DataFrame(mal_parts)
    assert len(mal_info) == 120 and len(mal) == 36848

    # ---------------- write
    np.save(os.path.join(args.out, "train_rows.npy"), tr)
    np.save(os.path.join(args.out, "train_segments.npy"), tr_seg)
    np.save(os.path.join(args.out, "benign_rows.npy"), ben)
    np.save(os.path.join(args.out, "benign_segments.npy"), ben_info[["start", "stop"]].to_numpy(np.int64))
    ben_info.to_csv(os.path.join(args.out, "benign_recordings.csv"), index_label="recording")
    np.save(os.path.join(args.out, "malicious_rows.npy"), mal)
    np.save(os.path.join(args.out, "malicious_segments.npy"), mal_info[["start", "stop"]].to_numpy(np.int64))
    mal_info.to_csv(os.path.join(args.out, "malicious_recordings.csv"), index_label="recording")
    shutil.copy2(os.path.join(LA, "scaler.json"), os.path.join(args.out, "scaler.json"))
    refit_on_all_rows(args.out)

    # ---------------- window counts per horizon
    val_seg = ben_info.loc[ben_info.role == "validation", ["start", "stop"]].to_numpy(np.int64)
    bt_seg = ben_info.loc[ben_info.role == "test", ["start", "stop"]].to_numpy(np.int64)
    mal_seg = mal_info[["start", "stop"]].to_numpy(np.int64)
    counts = {}
    for H in HORIZONS:
        W = INPUT_LENGTH + H
        counts[str(H)] = dict(
            train_all=int(len(all_starts(tr_seg, W)[0])),
            train_kept=int(len(thinned_starts(tr_seg, W)[0])),
            validation=int(len(all_starts(val_seg, W)[0])),
            benign_test=int(len(all_starts(bt_seg, W)[0])),
            malicious_test=int(len(all_starts(mal_seg, W)[0])),
            malicious_recordings_used=int(((mal_seg[:, 1] - mal_seg[:, 0]) >= W).sum()),
        )
    manifest = dict(
        source_csv="combined_labelled_25_model_features_20260929.csv",
        source_csv_sha256="4a20aedfdf103a22cf0ad9874b48087e82fe1179ea33bb26d230abccd27fce03",
        built_with="OWLYSHIELD_CRAFT_REPRODUCTION_v1: owly_reproduce_data.sh, then prepare_owly_forecast.py",
        features=json.load(open(os.path.join(LA, "scaler.json")))["feature_names"],
        scaling="StandardScaler refitted on train + test rows (every row in data/)",
        input_length=INPUT_LENGTH, horizons=list(HORIZONS),
        train_rule="full benign training recordings (808,637 rows); window starts thinned as CRAFT: blocks of 100, keep 1 in 16",
        validation_rule=f"{int(VAL_FRACTION*100)}% of benign test recordings per semantic label, seed {VAL_SEED}; all windows",
        benign_test_rule="remaining benign test recordings, whole; all windows (stride 1)",
        malicious_test_rule="the 120 attack recordings of CRAFT's medium test set (capped at 4,200 rows); all windows (stride 1)",
        recordings=dict(train=int(len(tr_seg)), validation=int(len(val_seg)), benign_test=int(len(bt_seg)),
                        malicious_test=int(len(mal_seg))),
        rows=dict(train=int(len(tr)), validation=int((val_seg[:, 1] - val_seg[:, 0]).sum()),
                  benign_test=int((bt_seg[:, 1] - bt_seg[:, 0]).sum()), malicious_test=int(len(mal))),
        windows=counts,
    )
    manifest["files"] = {f: sha256(os.path.join(args.out, f)) for f in sorted(os.listdir(args.out))
                         if f != "DATA_MANIFEST.json"}
    with open(os.path.join(args.out, "DATA_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in ("recordings", "rows", "windows")}, indent=2))


if __name__ == "__main__":
    main()
