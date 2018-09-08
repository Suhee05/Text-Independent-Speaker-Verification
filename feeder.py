import tensorflow as tf
import numpy as np
import time
from _thread import start_new_thread
import queue
import vad_ex
from python_speech_features import logfbank
import utils
import re
import os
import random
import pickle
import glob
import sys
import webrtcvad
import argparse


"""
input dir

vox1_dev_wav - id #### - 0DOmwbPlPvY - 00001.wav
                                     - 00002.wav
                                     - ...
                       - 5VNK93duiOM
                       - ...
                       
             - id #### - ...

"""

class Feeder():
    def __init__(self, hparams, data_type=None):
        # Set hparams
        self.hparams = hparams
        if self.hparams.mode == "train":
            assert data_type != None
            self.data_type = data_type

        # 나중에 좀 더 다양한 데이터를 input으로 받아서 process할 수 있도록 하는 부분 추가
    def set_up_feeder(self, queue=None):

        if self.hparams.mode == "train":
            #pickles = ["id11251_gFfcgOVmiO0_00004.pickle", "id11251_gFfcgOVmiO0_00005.pickle"...]
            self.pickles = os.listdir(self.hparams.in_dir + "/" + self.data_type)
            self.spk_names = list(set([pickle.split("_")[0] for pickle in self.pickles]))
            # Create Queue
            #self.queue = queue.Queue()
            self.queue = queue
            # Start new thread
            start_new_thread(self.generate_data, ())

        elif self.hparams.mode == "infer":
            #save_dict for saving mel spectrograms of two waves
            self.save_dict = {};

        elif self.hparams.mode == "test":
            pass
        else:
            raise ValueError("mode not supported")


    def generator(self, list, num_elements):
        # python gets list arg as reference
        batch = list[:num_elements]
        del list[:num_elements]
        list += batch
        return batch

    
    def is_invalid_spk(self, spk_id):
        # check if each speaker has more than at least self.hparams.num_utt_per_batch utterances
        spk_utt = [1 for pickle in self.pickles if re.search(spk_id, pickle)]
        num_utt = sum(spk_utt)
        if num_utt < self.hparams.num_utt_per_batch:
            return True
        else:
            return False

    def generate_data(self):
        while True:
            if self.queue.qsize() > 10:
                time.sleep(0.1)
                continue;

            #for i in range(self.num_batch):
                # generate a new batch and add it to the queue
            in_batch, target_batch = self.create_train_batch()

            self.queue.put([in_batch, target_batch])

        self.queue.task_done()
    
    def create_train_batch(self):
        num_frames = int(self.hparams.segment_length * 100)
        spk_batch = self.generator(self.spk_names, self.hparams.num_spk_per_batch)
        target_batch = [spk for spk in range(self.hparams.num_spk_per_batch) for i in range(self.hparams.num_utt_per_batch)]
        #print("spk_batch: " + str(spk_batch))
        #print("target_batch: " + str(target_batch))
        in_batch = []

        for spk_id in spk_batch:
            if self.is_invalid_spk(spk_id):
                print("speaker id: " + spk_id + " has less than " + str(self.hparams.num_utt_per_batch) + " utt files")
                continue
            # speaker_pickle_files_list ['id10645_xG_tys7Wrxg_00003.pickle', 'id10645_xG_tys7Wrxg_00004.pickle'...]
            speaker_pickle_files_list = [file_name for file_name in os.listdir(self.hparams.in_dir + "/" + self.data_type) if re.search(spk_id, file_name) is not None]
            num_pickle_per_speaker = len(speaker_pickle_files_list)

            # list of indices in speaker_pickle_files_list
            utt_idx_list = random.choices(range(num_pickle_per_speaker), k=self.hparams.num_utt_per_batch)
            #print("utt_idx_list for " +str(spk_id)+" is " + str(utt_idx_list))
            for utt_idx in utt_idx_list:
                utt_pickle = speaker_pickle_files_list[utt_idx]
                utt_path = self.hparams.in_dir + "/" + self.data_type + "/" + utt_pickle
                with open(utt_path, "rb") as f:
                    load_dict = pickle.load(f)
                    total_logmel_feats = load_dict["LogMel_Features"]

                # random start point for every utterance
                start_idx = random.randrange(0, total_logmel_feats.shape[0] - num_frames)
                #print("start index:" + str(start_idx))

                # total logmel_feats is a numpy array of [num_frames, nfilt]
                # slice logmel_feats by 160 frames (apprx. 1.6s) results in [160, 40] np array
                logmel_feats = total_logmel_feats[start_idx:start_idx+num_frames, :]
                in_batch.append(logmel_feats)

        in_batch = np.asarray(in_batch)
        target_batch = np.asarray(target_batch)

        return in_batch, target_batch

    def create_infer_batch(self):
        # self.hparams.in_wav1, self.hparams.in_wav2 are full paths of the wav file
        # for ex) /home/hdd2tb/ninas96211/dev_wav_set/id10343_pCDWKHjQjso_00002.wav

        wavs_list = [self.hparams.in_wav1, self.hparams.in_wav2]

        # file_name for ex) id10343_pCDWKHjQjso_00002
        for wav_path in wavs_list:
            wav_id = os.path.splitext(os.path.basename(wav_path))[0]
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
            logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=40)
            # file_name for ex, 'id10343_pCDWKHjQjso_00002'
            self.save_dict[wav_id] = logmel_feats

        num_frames = self.hparams.segment_length * 100
        num_overlap_frames = num_frames * self.hparams.overlap_ratio
        dvector_dict = {}

        match = False
        prev_wav_name = ""

        for wav_name, feats in self.save_dict.items():
            if wav_name.split("_")[0] == prev_wav_name:
                print("spk_id" + wav_name.split("_")[0])
                match = True
            total_len = feats.shape[0]
            num_dvectors = int((total_len - num_overlap_frames) // (num_frames - num_overlap_frames))
            print("num dvec:" + str(num_dvectors))
            dvectors = []
            for dvec_idx in range(num_dvectors):
                start_idx = int((num_frames - num_overlap_frames) * dvec_idx)
                end_idx = int(start_idx + num_frames)
                print("wavname: " + wav_name + " start_idx: " + str(start_idx) )
                print("wavname: " + wav_name + " end_idx: " + str(end_idx) )
                dvectors.append(feats[start_idx:end_idx, :])
            dvectors = np.asarray(dvectors)
            dvector_dict[wav_name] = dvectors
            prev_wav_name = wav_name.split("_")[0]


        wav1_data = list(dvector_dict.values())[0]
        wav2_data = list(dvector_dict.values())[1]

        print("match: " + str(match))
        print("wav1_data.shape:" + str(wav1_data.shape))
        print("wav2_data.shape:" + str(wav2_data.shape))
        return wav1_data, wav2_data, match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--in_dir", type=str, required=True, help="input data dir")
    args = parser.parse_args()
    feeder = Feeder(args)
    feeder.preprocess()
