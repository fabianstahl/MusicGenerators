# Comparing sample-based music generation algorithms using deep learning

This work was made during a project in summer 2019.

Please have a look at the documentation: **Doc/Doc/doc.pdf** 

Fabian Stahl, University of applied Science, 65185 Wiesbaden

# SampleRNN

SampleRNN Paper: <https://arxiv.org/abs/1612.07837>

## Training

```
usage: train.py [-h] --exp EXP --frame_sizes FRAME_SIZES [FRAME_SIZES ...]
                --dataset DATASET [--n_rnn N_RNN] [--dim DIM]
                [--learn_h0 LEARN_H0] [--q_levels Q_LEVELS]
                [--seq_len SEQ_LEN] [--weight_norm WEIGHT_NORM]
                [--batch_size BATCH_SIZE] [--val_frac VAL_FRAC]
                [--test_frac TEST_FRAC]
                [--keep_old_checkpoints KEEP_OLD_CHECKPOINTS]
                [--datasets_path DATASETS_PATH] [--results_path RESULTS_PATH]
                [--epoch_limit EPOCH_LIMIT] [--resume RESUME]
                [--sample_rate SAMPLE_RATE] [--n_samples N_SAMPLES]
                [--sample_length SAMPLE_LENGTH]
                [--loss_smoothing LOSS_SMOOTHING] [--cuda CUDA] [--gpu GPU]
                [--dropout DROPOUT] [--lr LR] [--load_model LOAD_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             experiment name
  --frame_sizes FRAME_SIZES [FRAME_SIZES ...]
                        frame sizes in terms of the number of lower tier
                        frames, starting from the lowest RNN tier
  --dataset DATASET     dataset name - name of a directory in the datasets
                        path (settable by --datasets_path)
  --n_rnn N_RNN         number of RNN layers in each tier (default: 3)
  --dim DIM             number of neurons in every RNN and MLP layer (default:
                        1024)
  --learn_h0 LEARN_H0   whether to learn the initial states of RNNs (default:
                        True)
  --q_levels Q_LEVELS   number of bins in quantization of audio samples
                        (default: 256)
  --seq_len SEQ_LEN     how many samples to include in each truncated BPTT
                        pass (default: 1024)
  --weight_norm WEIGHT_NORM
                        whether to use weight normalization (default: True)
  --batch_size BATCH_SIZE
                        batch size (default: 128)
  --val_frac VAL_FRAC   fraction of data to go into the validation set
                        (default: 0.1)
  --test_frac TEST_FRAC
                        fraction of data to go into the test set (default:
                        0.1)
  --keep_old_checkpoints KEEP_OLD_CHECKPOINTS
                        whether to keep checkpoints from past epochs (default:
                        False)
  --datasets_path DATASETS_PATH
                        path to the directory containing datasets (default:
                        datasets)
  --results_path RESULTS_PATH
                        path to the directory to save the results to (default:
                        results)
  --epoch_limit EPOCH_LIMIT
                        how many epochs to run (default: 100)
  --resume RESUME       whether to resume training from the last checkpoint
                        (default: False)
  --sample_rate SAMPLE_RATE
                        sample rate of the training data and generated sound
                        (default: 16000)
  --n_samples N_SAMPLES
                        number of samples to generate in each epoch (default:
                        1)
  --sample_length SAMPLE_LENGTH
                        length of each generated sample (in samples) (default:
                        80000)
  --loss_smoothing LOSS_SMOOTHING
                        smoothing parameter of the exponential moving average
                        over training loss, used in the log and in the loss
                        plot (default: 0.99)
  --cuda CUDA           whether to use CUDA (default: True)
  --gpu GPU             which GPU to use (default: 1)
  --dropout DROPOUT     dropout probability (default: 0.0)
  --lr LR               learning rate (default: 0.001)
  --load_model LOAD_MODEL
                        Load a certain model with this path
```


# VRNN

VRNN Paper: <https://arxiv.org/abs/1506.02216>

## Training

```
usage: train.py [-h] [--gpu GPU] [--lr LR] [--clip CLIP]
                [--batch_size BATCH_SIZE] [--x_dim X_DIM] [--z_dim Z_DIM]
                [--n_rnn_layers N_RNN_LAYERS] [--epochs EPOCHS]
                [--save_interval SAVE_INTERVAL] [--test_seq_len TEST_SEQ_LEN]
                [--x_f_dim X_F_DIM] [--z_f_dim Z_F_DIM] [--enc_dim ENC_DIM]
                [--dec_dim DEC_DIM] [--h_dim H_DIM] [--prior_dim PRIOR_DIM]
                [--sample_rate SAMPLE_RATE] --dataset DATASET --output_dir
                OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             which GPU to use (default: 1)
  --lr LR               learning rate (default: 0.0003)
  --clip CLIP           clip value for gradient clipping (default: 0.5)
  --batch_size BATCH_SIZE
                        the batch size (default: 128)
  --x_dim X_DIM         number of input sample per window (default: 200)
  --z_dim Z_DIM         dimension of the latent variable z (default: 200)
  --n_rnn_layers N_RNN_LAYERS
                        number of stacked RNN layers (default: 1)
  --epochs EPOCHS       stop training after this amount of epochs (default:
                        10000)
  --save_interval SAVE_INTERVAL
                        save the model in this interval (default: 10)
  --test_seq_len TEST_SEQ_LEN
                        length of test samples after each epoch (default: 300)
  --x_f_dim X_F_DIM     number of intermediate features in phi-x (default:
                        600)
  --z_f_dim Z_F_DIM     number of intermediate features in phi-z (default:
                        500)
  --enc_dim ENC_DIM     number of intermediate features in the encoder
                        (default: 500)
  --dec_dim DEC_DIM     number of intermediate features in the decoder
                        (default: 600)
  --h_dim H_DIM         dimension of state vector h (default: 2000)
  --prior_dim PRIOR_DIM
                        number of intermediate features in the prior (default:
                        500)
  --sample_rate SAMPLE_RATE
                        sample rate of the used music (default: 16000)
  --dataset DATASET     name of a prepaired folder with music snippets as .wav
                        or .mp3)
  --output_dir OUTPUT_DIR
                        output directory for saved graphs, samples and
                        tensorboard logs
```



# WaveGAN
WaveGAN Paper: <https://arxiv.org/abs/1802.04208>
## Training
```
usage: train.py [-h] [--gpu GPU] [--lr LR] [--alpha ALPHA]
                [--phaseshuffle_n PHASESHUFFLE_N] [--batch_size BATCH_SIZE]
                [--d D] [--channels CHANNELS] [--samples SAMPLES]
                [--steps STEPS] [--dis_up_per_gen_up DIS_UP_PER_GEN_UP]
                [--mom1 MOM1] [--mom2 MOM2] [--save_interval SAVE_INTERVAL]
                [--generate_interval GENERATE_INTERVAL] [--lambda LAMBDA]
                [--sample_rate SAMPLE_RATE] --dataset DATASET --output_dir
                OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             which GPU to use (default: 1)
  --lr LR               learning rate (default: 0.0001)
  --alpha ALPHA         Leaky Relu factor (default: 0.2)
  --phaseshuffle_n PHASESHUFFLE_N
                        maximal shuffeling offset in discriminator (default:
                        2)
  --batch_size BATCH_SIZE
                        the batch size (default: 64)
  --d D                 the models scale factor (default: 64)
  --channels CHANNELS   number of audio channels (default: 1)
  --samples SAMPLES     number of input / output samples, only 16384 and 65536
                        allowed (default: 65536)
  --steps STEPS         stop training after this amount of training steps
                        (default: 1000000)
  --dis_up_per_gen_up DIS_UP_PER_GEN_UP
                        discriminator updates per generator update (default:
                        5)
  --mom1 MOM1           Adam optimizer beta1 (default: 0.5)
  --mom2 MOM2           Adam optimizer beta2 (default: 0.9)
  --save_interval SAVE_INTERVAL
                        save the model in this interval (default: 1000)
  --generate_interval GENERATE_INTERVAL
                        generates samples in this interval (default: 100)
  --lambda LAMBDA       lambda value of gradient penalty (default: 10)
  --sample_rate SAMPLE_RATE
                        the sample rate (samples per second) (default: 16000)
  --dataset DATASET     name of a prepaired .data numpy arraw (using
                        prepaire_Music_chunks.py))
  --output_dir OUTPUT_DIR
                        output directory for saved graphs, samples and
                        tensorboard logs
```


