import tensorflow as tf
import re
import os
import glob
import sys
import pickle
import random
import numpy as np
import argparse
import time
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import utils
import vad_ex
import webrtcvad


"""
input dir

vox1_dev_wav - id #### - 0DOmwbPlPvY - 00001.wav
                                     - 00002.wav
                                     - ...
                       - 5VNK93duiOM
                       - ...
                       
             - id #### - ...

"""

def main():

	# Hyperparameters

    parser = argparse.ArgumentParser()

    # in_dir = ~/wav
    parser.add_argument("--in_dir", type=str, required=True, help="input audio data dir")
    parser.add_argument("--data_type", required=True, choices=["libri", "vox1", "vox2"])

    # Data Process
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")
    args = parser.parse_args()

    pk_dir = os.path.dirname(args.in_dir.rstrip("/")) + "/wavs_pickle"

    # try to make pickle directory.
    try:
    	os.mkdir(pk_dir)
    	print("pickle directory created.")
    except FileExistsError:
    	print("wavs_pickle already exists.")
    except:
    	print("Unexpected Error:", sys.exc_info()[0])

    if args.data_type == "vox1":
    	# full path of all audio files in in_dir
    	wavs = glob.iglob(args.in_dir.rstrip("/")+"/*/*/*.wav")

    	for wav_path in wavs:

		    print(wav_path)
		    wav_id = "_".join(wav_path.split("/")[-3:])

		    # VAD Process
		    audio, sample_rate = vad_ex.read_wave(wav_path)
		    vad = webrtcvad.Vad(1)
		    frames = vad_ex.frame_generator(30, audio, sample_rate)
		    frames = list(frames)
		    segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
		    total_wav = b""
		    for i, segment in enumerate(segments):
		        total_wav += segment
		        print(wav_id+ " : " + str(i)+"th segment appended")

		    # Without writing, unpack total_wav into numpy [N,1] array
		    # 16bit PCM 기준 dtype=np.int16
		    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
		    print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
		    
		    # if wav is smaller than 1.6s, throw away
		    if round((wav_arr.shape[0] / sample_rate), 1) > args.segment_length:
		        logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=args.spectrogram_scale)
		        print("created logmel feats from audio data. np array of shape:"+str(logmel_feats.shape))
		        save_dict = {};
		        save_dict["SpkId"] = wav_path.split("/")[-3]
		        save_dict["ClipId"] = wav_path.split("/")[-2]
		        save_dict["WavId"] = wav_path.split("/")[-1]
		        save_dict["LogMel_Features"] = logmel_feats;
		        pickle_f_name = wav_id.replace("wav", "pickle")
		        with open(pk_dir + "/" + pickle_f_name, "wb") as f:
		            pickle.dump(save_dict, f, protocol=3);
		    else:
		        print("wav length smaller than 1.6s: " + wav_id)




if __name__ == "__main__":
    main()