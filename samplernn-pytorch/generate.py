import os
import re
import subprocess
import sys


command = "python3 train.py --exp {} --frame_sizes {} --n_rnn {} --dataset {} --gpu 0 --resume True --sample_length 1280000 --epoch_limit 0 --results_path songs --n_samples 5 --load_model {}"

#print(sorted(os.listdir("results")))
#sys.exit(-1)
for exp in sorted(os.listdir("results"))[14:]:
    checkpoint_path = os.path.join("results", exp, "checkpoints")
    if not os.path.exists(checkpoint_path):
        continue
    
    print(checkpoint_path)
    

    n_rnn = 3
    frame_sizes = "16 4"
    for cmd in exp.split('-'):
        if cmd.split(':')[0]=='dataset':
            dataset = cmd.split(':')[1]
        if cmd.split(':')[0]=='n_rnn':
            n_rnn = cmd.split(':')[1]
        if cmd.split(':')[0]=='frame_sizes':
            frame_sizes = cmd.split(':')[1].replace(',',' ')
        if cmd.split(':')[0]=='exp':
            exp = cmd.split(':')[1]

    match_path = sorted(os.listdir(checkpoint_path))[-2]
    model_path = os.path.join(checkpoint_path, match_path)

    cmd = command.format(exp, frame_sizes, n_rnn, dataset, model_path)
    print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Finished "+exp)
    #break
