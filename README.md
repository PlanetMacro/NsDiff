python3 src/experiments/pretrain_f.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 runs --seeds='[1216]' 


python3 src/experiments/pretrain_g.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 --rolling_length=48 runs --seeds='[1216]' 

python3 src/experiments/NsDiff.py --dataset_type="ETTh1" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 --rolling_length=48 --load_pretrain=True runs --seeds='[1216]'

# Custom CSV Dataset

To use a custom time series CSV file, place your file in the `data/custom/` directory. The CSV should contain a datetime column first, followed by one or more feature columns. Then run:
```
python3 src/experiments/NsDiff.py --dataset_type="CustomCsv" --device="cuda:0" --batch_size=32 --horizon=1 --pred_len=192 --windows=168 --epochs=20 --patience=5 --num_worker=0 runs
```

# Custom Multi-CSV Dataset

To use multiple time series CSV files simultaneously, place your CSV files in the `data/custom/` directory. Each CSV file should have a datetime column first, followed by one or more feature columns. Filenames can be arbitrary.

Then run the experiment with:
```
python3 src/experiments/NsDiff.py \
  --dataset_type="custom_multy" \
  --device="cuda:0" \
  --batch_size=32 \
  --horizon=1 \
  --pred_len=192 \
  --windows=168 \
  --epochs=20 \
  --patience=5 \
  --num_worker=0 \
  runs
```

Note: the `--dataset_type` flag is case-sensitive and must be set to `custom_multy` to use the multi-CSV loader.

