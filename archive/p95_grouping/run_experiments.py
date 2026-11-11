from pathlib import Path
import subprocess,sys,json
O=Path(__file__).resolve().parent;R=O.parents[1]
args='--mode ablate --epochs 360 --patience 80 --batch-size 512 --dropout 0.02 --label-smoothing 0 --learning-rate 0.0015 --weight-decay 1e-5 --adam-beta2 0.99 --optimizer adamw --scheduler cosine --device cuda --features raw5 --architectures residual32_gelu_relu_hybrid --head-scale 20 --hybrid-alpha-init 0.5 --checkpoint-average-k 5 --loss ala_linear --ldam-max-margin 0.3 --ala-linear-strength 0.2 --ala-schedule warmup_decay --ala-warmup-fraction 0.15 --ala-peak-fraction 0.65 --ala-final-factor 0.15 --pairwise-ala-strength 0.25 --trajectory-consistency-weight 0.05 --physics-teacher lr --physics-per-point 8 --physics-sigma 0.06 --physics-weight 1.0 --physics-batch-ratio 1.0 --physics-mode jitter --physics-weight-schedule cosine_decay --physics-final-weight 0.0 --physics-decay-end 0.7 --physics-min-distance 0.05 --protocols sweep_group --seeds 42 43 44 45 46'.split()
data=['--x',str(R/'esp32p4_iat_demo/input/data/x_train.csv'),'--y',str(R/'esp32p4_iat_demo/input/data/y_train.csv')]
cmd=[sys.executable,str(O/'train_p95.py'),*args,*data,'--output',str(O/'training'),'--ablate-export-dir',str(O/'models')]
(O/'fp32_command.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2),encoding='utf8')
with (O/'fp32.log').open('w',encoding='utf8') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
for seed in range(42,47):
    cmd=[sys.executable,str(O/'source_snapshot/qat_A.py'),str(O/f'models/sweep_group_seed{seed}'),*data,'--device','cuda','--selection','accuracy','--out-name','int8_A']
    with (O/f'qat_{seed}.log').open('w',encoding='utf8') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
(O/'TRAINING_COMPLETED.txt').write_text('Five FP32 and five INT8-A runs completed. Evaluation and paper revision still required.',encoding='utf8')
