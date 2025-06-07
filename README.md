python3 src/experiments/pretrain_f.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 runs --seeds='[1216]' 


python3 src/experiments/pretrain_g.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 --rolling_length=48 runs --seeds='[1216]' 

python3 src/experiments/NsDiff.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 --rolling_length=48 --load_pretrain=True runs --seeds='[1216]'
