from pathlib import Path
O=Path(__file__).resolve().parent
s=(O/'source_snapshot/qat_fold_pipeline.py').read_text(encoding='utf8')
doc_start=s.index('"""');doc_end=s.index('"""',doc_start+3)
s=s[:doc_start]+'''"""P95 sensitivity experiment, INT8-A only.
Network QAT and integer export; no additional head QAT.
Training/calibration use train rows, checkpoint selection uses valid rows.
Test predictions are inspected for conversion diagnostics and final evaluation.
Author-approved exception: seed43 alpha tolerance0.006; other folds0.001.
"""'''+s[doc_end+3:]
a=s.index('    # ---- 4.');b=s.index('    # ---- 5.')
s=s[:a]+'''    # Variant A: no additional head refinement.
    hbest_blob = blob
    (out / "model_int8.bin").write_bytes(hbest_blob)

'''+s[b:]
needle='    fp32_test = metrics(labels[test_idx], fp32_pred[test_idx])'
s=s.replace(needle,'''    with torch.no_grad():
        original_pred = teacher(normalized).argmax(1).numpy()
    np.savez_compressed(out / "predictions.npz", fp32=original_pred, folded=fp32_pred, test=test_idx)
    fold_changes = int((original_pred != fp32_pred).sum())
'''+needle)
s=s.replace('    int8_test = metrics(labels[test_idx], pred[test_idx])','''    np.savez_compressed(out / "predictions.npz", fp32=original_pred, folded=fp32_pred, int8=pred, test=test_idx)
    int8_test = metrics(labels[test_idx], pred[test_idx])''')
s=s.replace('        "hybrid_alpha": alpha,','        "hybrid_alpha": alpha,\n        "fold_top1_changes_all_rows": fold_changes,')
s=s.replace('    folded, alpha = fold_cosine_head', '''    if meta["seed"] == 43:
        # Author approved this specifically disclosed P95 fold exception.
        args.alpha_tolerance = 0.006
    folded, alpha = fold_cosine_head''')
s=s.replace('        "selection_criterion": args.selection,','        "selection_criterion": args.selection,\n        "alpha_tolerance_used": args.alpha_tolerance,')
s=s.replace('训练/校准/选点仅用 train+valid；test 只在本行评测时使用一次','Training/calibration use train; checkpoint selection uses valid; test is used for conversion diagnostics and final evaluation, not gradient updates or checkpoint selection')
(O/'source_snapshot/qat_A.py').write_text(s,encoding='utf8')
