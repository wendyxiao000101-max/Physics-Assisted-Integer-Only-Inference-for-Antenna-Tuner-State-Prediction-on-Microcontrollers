"""Recompute Figure 7 data from the included original experiment logs.
Run with Python + numpy + pandas; no training, synthetic data or smoothing.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = range(42, 47)
ARMS = ['A0_none', 'B0_clean', 'B2_eta015', 'C_shuffle']
LABELS = {'A0_none':'No physics', 'B0_clean':'Original teacher',
          'B2_eta015':'Coefficient noise eta=0.15', 'C_shuffle':'Frequency shuffle'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def csv(frame, name):
    frame.to_csv(ROOT/name, index=False, encoding='utf-8-sig')

def main():
    manifest = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        assert sha(ROOT/item['package_path']) == item['sha256'], item['package_path']
    y = pd.read_csv(ROOT/'sources/dataset/y_train.csv').to_numpy().ravel().astype(int)
    supports = {}; seen = np.zeros(len(y), dtype=int)
    for seed in SEEDS:
        with np.load(ROOT/f'sources/splits/sweep_group_seed{seed}/split.npz') as s:
            train, valid, test = s['train'], s['valid'], s['test']
            assert not set(test)&set(train) and not set(test)&set(valid)
            assert len(np.unique(test)) == len(test)
            supports[seed] = np.bincount(y[test], minlength=15)
            seen[test] += 1
    assert len(y) == 96768 and np.all(seen == 1)
    records = []
    for arm in ARMS:
        base = ROOT/'sources/teacher_gain'/arm
        runs = pd.read_csv(base/'raw_runs.csv')
        assert len(runs) == 5 and set(runs.seed) == set(SEEDS)
        assert (runs.protocol == 'sweep_group').all()
        for seed in SEEDS:
            filename = f'raw5__residual32_gelu_relu_hybrid__sweep_group__{seed}.json'
            obj = json.loads((base/'runs'/filename).read_text(encoding='utf-8'))
            cm = np.asarray(obj['test']['confusion_matrix'], dtype=int)
            assert cm.shape == (15, 15) and (cm >= 0).all()
            assert np.array_equal(cm.sum(axis=1), supports[seed])
            n = int(cm.sum()); correct = int(np.trace(cm)); acc = correct/n
            row = runs.loc[runs.seed == seed].iloc[0]
            assert int(obj['summary']['seed']) == seed
            assert obj['summary']['protocol'] == 'sweep_group'
            np.testing.assert_allclose([row.test_accuracy, obj['test']['accuracy'],
                                       obj['summary']['test_accuracy']], acc, atol=1e-14, rtol=0)
            records.append(dict(arm=arm,condition=LABELS[arm],fold=seed-41,seed=seed,
                                test_rows=n,test_correct=correct,student_accuracy_pct=100*acc,
                                source_log=str(Path('sources/teacher_gain')/arm/'runs'/filename)))
    detail = pd.DataFrame(records)
    wideacc = detail.pivot(index='seed', columns='arm', values='student_accuracy_pct')
    paired = []; summary = []; coordinates = []
    reference = pd.read_csv(ROOT/'sources/reference/T21_teacher_quality_vs_student_gain.csv').set_index('arm')
    offsets = np.linspace(-.16,.16,5);markers = ['o','s','^','v','P']
    for category, arm in enumerate(ARMS[1:]):
        gains = wideacc[arm]-wideacc['A0_none']
        mean, sd = gains.mean(), gains.std(ddof=1)
        np.testing.assert_allclose([mean,sd],reference.loc[arm,['gain_pp','gain_std_pp']].to_numpy(dtype=float),atol=1e-10,rtol=0)
        for seed in SEEDS:
            active = detail[(detail.arm==arm)&(detail.seed==seed)].iloc[0]
            base = detail[(detail.arm=='A0_none')&(detail.seed==seed)].iloc[0]
            paired.append(dict(arm=arm,condition=LABELS[arm],fold=seed-41,seed=seed,
                               test_rows=int(active.test_rows),baseline_correct=int(base.test_correct),
                               student_correct=int(active.test_correct),baseline_accuracy_pct=wideacc.loc[seed,'A0_none'],
                               student_accuracy_pct=wideacc.loc[seed,arm],gain_pp=gains.loc[seed]))
            coordinates.append(dict(arm=arm,condition=LABELS[arm],category_x=category,fold=seed-41,seed=seed,
                                    scatter_x=category+offsets[seed-42],gain_pp=gains.loc[seed],marker=markers[seed-42]))
        summary.append(dict(arm=arm,condition=LABELS[arm],category_x=category,mean_marker_x=category+.29,
                            n_pairs=5,mean_gain_pp=mean,sample_sd_gain_pp=sd,
                            lower_mean_minus_sd_pp=mean-sd,upper_mean_plus_sd_pp=mean+sd,positive_folds=int((gains>0).sum())))
    paired=pd.DataFrame(paired);summary=pd.DataFrame(summary)
    oldpaired=pd.read_csv(ROOT/'sources/reference/figure1_paired_gains.csv')
    np.testing.assert_allclose(paired[['baseline_accuracy_pct','student_accuracy_pct','gain_pp']],
                               oldpaired[['baseline_accuracy_pct','student_accuracy_pct','gain_pp']],atol=1e-10,rtol=0)
    assert np.array_equal(paired[['arm','fold','seed']],oldpaired[['arm','fold','seed']])
    oldsummary=pd.read_csv(ROOT/'sources/reference/figure1_gain_summary.csv')
    np.testing.assert_allclose(summary[['mean_gain_pp','sample_sd_gain_pp']],oldsummary[['mean_gain_pp','sample_sd_gain_pp']],atol=1e-10,rtol=0)
    olddetail=pd.read_csv(ROOT/'sources/reference/figure1_accuracy_per_arm_fold.csv')
    assert np.array_equal(detail[['test_rows','test_correct']],olddetail[['test_rows','test_correct']])
    wide = pd.DataFrame({'fold':list(range(1,6)),'seed':list(SEEDS)})
    for arm,name in [('B0_clean','Original_teacher_gain_pp'),('B2_eta015','Coefficient_noise_eta015_gain_pp'),('C_shuffle','Frequency_shuffle_gain_pp')]:
        wide[name] = paired.loc[paired.arm==arm,'gain_pp'].to_numpy()
    csv(wide,'Figure7_fold_gains_wide.csv')
    csv(paired,'Figure7_paired_gains_15_records.csv')
    csv(summary,'Figure7_mean_SD_errorbars.csv')
    csv(detail,'Figure7_accuracy_20_runs.csv')
    csv(pd.DataFrame(coordinates),'Figure7_scatter_coordinates.csv')
    back=pd.read_csv(ROOT/'Figure7_fold_gains_wide.csv')
    np.testing.assert_allclose(back.iloc[:,2:],wide.iloc[:,2:],atol=1e-12,rtol=0)
    verification=dict(source_files_hash_verified=len(manifest),original_log_runs_recomputed=len(detail),
                      matched_gain_records=len(paired),original_test_records=len(y),test_folds_cover_each_record_once=True,
                      all_confusion_matrices_class_supports_match_original_splits=True,
                      gain_definition='100 * (student_correct - baseline_correct) / test_rows',
                      summary_definition='Unweighted mean and sample SD (ddof=1) of five paired fold gains',
                      matches_existing_plotting_exports=True,matches_T21=True,
                      summary=summary.to_dict(orient='records'),
                      limitation='Archived run logs contain confusion matrices and training histories, but no per-record student predictions. No retraining or inference rerun is claimed.')
    (ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['condition','mean_gain_pp','sample_sd_gain_pp','positive_folds']].to_string(index=False))
    print('Verified 20 runs, 15 fold gains, 3 mean/SD error bars.')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
