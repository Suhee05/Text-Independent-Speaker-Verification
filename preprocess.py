import tensorflow as tf
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

class Preprocess():
    def __init__(self, hparams, data_type):
        # Set hparams
        self.hparams = hparams
        self.data_type = data_type

        # Start Process
        start_new_thread(self.preprocess_data(), ())

    def preprocess_data(self):
        if self.data_type == "libri":
            path_list = glob.iglob(self.hparams.in_dir.rstrip("/")+"/*/*.wav")
        elif self.data_type == "vox1":
            path_list = glob.iglob(self.hparams.in_dir.rstrip("/")+"/*/*/*.wav")
        elif self.data_type == "vox2":
            path_list = glob.iglob(self.hparams.in_dir.rstrip("/")+"/*/*/*.m4a")
        else:
            raise ValueError("data type not supported")
        for path in path_list:
            print(path)
            wav_arr, sample_rate = self.vad_process(path)
            self.create_pickle(path, wav_arr, sample_rate)

    def vad_process(self, path):
        # VAD Process
        if self.data_type == "vox1":
            audio, sample_rate = vad_ex.read_wave(path)
        elif self.data_type == "vox2":
            audio, sample_rate = vad_ex.read_m4a(path)
        elif self.data_type == "libri":
            audio, sample_rate = vad_ex.read_libri(path)
        vad = webrtcvad.Vad(1)
        frames = vad_ex.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
        total_wav = b""
        for i, segment in enumerate(segments):
            total_wav += segment
        # Without writing, unpack total_wav into numpy [N,1] array
        # 16bit PCM 기준 dtype=np.int16
        wav_arr = np.frombuffer(total_wav, dtype=np.int16)
        print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
        return wav_arr, sample_rate

    def create_pickle(self, path, wav_arr, sample_rate):
        os.mkdir(self.hparams.pk_dir + "/" + self.data_type)
        if round((wav_arr.shape[0] / sample_rate), 1) > self.hparams.segment_length:
            save_dict = {};
            logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=self.hparams.spectrogram_scale)
            print("created logmel feats from audio data. np array of shape:"+str(logmel_feats.shape))
            save_dict["LogMel_Features"] = logmel_feats;

            if self.data_type == ("vox1" or "vox2"):
            	data_id = "_".join(path.split("/")[-3:])
            	save_dict["SpkId"] = path.split("/")[-3]
            	save_dict["ClipId"] = path.split("/")[-2]
            	save_dict["WavId"] = path.split("/")[-1]
            	if self.data_type == "vox1":
            	    pickle_f_name = data_id.replace("wav", "pickle")
            	elif self.data_type == "vox2":
            	    pickle_f_name = data_id.replace("m4a", "pickle")

            elif self.data_type == "libri":
                data_id = "_".join(path.split("/")[-2:])
                save_dict["SpkId"] = path.split("/")[-2]
                save_dict["WavId"] = path.split("/")[-1]
                pickle_f_name = data_id.replace("wav", "pickle")
                print(pickle_f_name)

            with open(self.hparams.pk_dir + "/" + self.data_type + "/" + pickle_f_name, "wb") as f:
                pickle.dump(save_dict, f, protocol=3);
        else:
            print("wav length smaller than 1.6s: " + path)

def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # in_dir = ~/wav
    parser.add_argument("--in_dir", type=str, required=True, help="input audio data dir")
    parser.add_argument("--pk_dir", type=str, required=True, help="output pickle dir")
    #parser.add_argument("--data_type", required=True, choices=["libri", "vox1", "vox2"])

    # Data Process
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")
    args = parser.parse_args()

    #pk_dir = os.path.dirname(args.in_dir.rstrip("/")) + "/wavs_pickle"

    # try to make pickle directory.
    try:
    	os.mkdir(args.pk_dir)
    	print("pickle directory created.")
    except FileExistsError:
    	print("wavs_pickle already exists.")
    except:
    	print("Unexpected Error:", sys.exc_info()[0])


    libri_preprocess = Preprocess(args, "libri")
    #libri_preprocess.preprocess_data()

    vox1_preprocess = Preprocess(args, "vox1")
    #vox1_preprocess.preprocess_data()

    vox2_preprocess = Preprocess(args, "vox2")
    #vox2_preprocess.preprocess_data()
    



if __name__ == "__main__":
    main()