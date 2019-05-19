dataset=$1
python3 train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset $dataset --gpu 1 --resume True --sample_length 1280000 --epoch_limit 0 --results_path songs --n_samples
