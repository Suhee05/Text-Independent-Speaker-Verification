# Text Independant Speaker Verification

Tensorflow implementation of Text Independant Speaker Verification based on [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) and [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)


## Data
Both papers above used internal data which consist of
36M utterances from 18K speakers.
The original data is substituted with the combination of VoxCeleb1,2 and LibriSpeech. All of them are available for free. 
The whole data of those 3 has 10% EER whereas the original one has 5% EER according to Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. 
Below are links for the data



[LibriSpeech](http://www.openslr.org/12/)

[VoxCeleb1,2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## Prerequisites
Use requirements.txt for installing python packages.

- Python
- Tensorflow-gpu 1.6.0
- NVIDIA GPU + CUDA 9.0 + CuDNN 7.0


## Training

### 1. Preprocess wav data into spectrogram

**Currently this model only supports voxceleb1.**
</br>
**Preprocess for LibriSpeech and Voxceleb2 will be uploaded soon**
</br>

+ Downloaded data has a tree structure like below

```
wav_root - speaker_id - video_clip_id - 00001.wav
                                      - 00002.wav
                                      - ...
                                      
```
+ Run preprocess.py


```python
python preprocess.py --in_dir /home/ninas96211/data/wav_root --data_type vox1
```

### 2. Train 

+ Run train.py

```python
python train.py --in_dir /home/ninas96211/data/wavs_pickle --ckpt_dir ./ckpt
```

### 3. Infer

+ Using data\_gen.sh, create a directory for test where wavs have names like [speaker\_id]\_[video\_clip\_id]\_[wav\_number].wav


```bash
bash data_gen.sh /home/ninas96211/data/test_wav/id10275/CVUXDNZzcmA/00002.wav ~/data/test_wav_set
```

+ Run inference.py

```python
python inference.py --in_wav1 /home/ninas96211/data/test_wav_set/id10309_pwfqGqgezH4_00004.wav --in_wav2 /home/ninas96211/data/test_wav_set/id10296_f_k09R8r_cA_00004.wav --ckpt_file ./ckpt/model.ckpt-35000
```