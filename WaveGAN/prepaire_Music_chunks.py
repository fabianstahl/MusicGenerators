import librosa
import numpy as np
import pickle
import os


#root_path = "./data/drums"
root_path = "./data/mendelson"
compressed_folder = "./raw"
fs = 16000
block_size = 65536 #16384
extensions = [".wav", ".mp3"]
files = [os.path.join(root_path, im_path) for im_path in sorted(os.listdir(root_path)) if os.path.splitext(im_path)[1] in extensions]




def decode_audio(fp, fs=None, num_channels=1, normalize=False, fast_wav=False):
        """Decodes audio file paths into 32-bit floating point vectors.

        Args:
            fp: Audio file path.
            fs: If specified, resamples decoded audio to this rate.
            mono: If true, averages channels to mono.
            fast_wav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).

        Returns:
            A np.float32 array containing the audio samples at specified sample rate.
        """
        if fast_wav:
            # Read with scipy wavread (fast).
            _fs, _wav = wavread(fp)
            if fs is not None and fs != _fs:
                raise NotImplementedError('Scipy cannot resample audio.')
            if _wav.dtype == np.int16:
                _wav = _wav.astype(np.float32)
                _wav /= 32768.
            elif _wav.dtype == np.float32:
                _wav = np.copy(_wav)
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
        else:
            # Decode with librosa load (slow but supports file formats like mp3).
            _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
            if _wav.ndim == 2:
                _wav = np.swapaxes(_wav, 0, 1)
        
            
        assert _wav.dtype == np.float32

        # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
        # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
        if _wav.ndim == 1:
            nsamps = _wav.shape[0]
            nch = 1
        else:
            nsamps, nch = _wav.shape
        _wav = np.reshape(_wav, [nch, nsamps])
        
        # Average (mono) or expand (stereo) channels
        if nch != num_channels:
            if num_channels == 1:
                _wav = np.mean(_wav, 0, keepdims=True)
            elif nch == 1 and num_channels == 2:
                _wav = np.concatenate([_wav, _wav], axis=0)
            else:
                raise ValueError('Number of audio channels not equal to num specified')

        if normalize:
            factor = np.max(np.abs(_wav))
            if factor > 0:
                _wav /= factor

        return np.array(_wav)


def pad(array, main_size):
    missing_elements = block_size - len(array)
    return np.concatenate([array, np.zeros(missing_elements).astype(np.float32)])


if not os.path.exists(compressed_folder):
    os.makedirs(compressed_folder)

chunks = []

for f in files:
    song_data = decode_audio(fp=f, fs=fs)[0]
    #print(song_data.shape)
    for x in range(0,len(song_data),block_size):

        chunk = song_data[x:x + block_size]
        if len(chunk) == block_size:
            chunks.append(chunk)
        else:
            chunk = pad(chunk, block_size)
            chunks.append(chunk)
            print("Finished File '{}'".format(f))

print("Split {} songs into {} chunks!".format(len(files), len(chunks)))

dataset = root_path.split("/")[-1]
print(root_path.split("/"))
new_path = os.path.join(compressed_folder, dataset + ".data")
print("Saving under {}".format(new_path))
with open(new_path, "wb") as f:
    pickle.dump(np.array(chunks), f)

