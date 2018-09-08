# Text Independant Speaker Verification Using GE2E Loss

Tensorflow implementation of Text Independent Speaker Verification based on [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) and [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)


## Data
Both papers above used internal data which consist of
36M utterances from 18K speakers.
In this repository, the original dataset was substituted with the combination of VoxCeleb1,2 and LibriSpeech. All of them are available for free. 
The whole data of those 3 have 10% EER whereas the original one has 5% EER according to Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis. 
Below are links for the data </br>
**Downloading data will be soon added to preprocess.py. Before that, manually download dataset using the links below**

[LibriSpeech](http://www.openslr.org/12/)

[VoxCeleb1,2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## Prerequisites
Use requirements.txt for installing python packages.

`
pip install -r requirements.txt
`

- Python
- Tensorflow-gpu 1.6.0
- NVIDIA GPU + CUDA 9.0 + CuDNN 7.0


## Training

### 1. Preprocess wav data into spectrogram


+ VoxCeleb1 each has a tree structure like below

```
wav_root - speaker_id - video_clip_id - 00001.wav
                                      - 00002.wav
                                      - ...
                                      
```
+ VoxCeleb2 each has a tree structure like below

```
wav_root - speaker_id - video_clip_id - 00001.m4a
                                      - 00002.m4a
                                      - ...
                                      
```
+ LibriSpeech has a tree structure like below </br>
**To have a tree structre below, LibriSpeech dataset has to be preprocessed before running preprocess.py. </br> Ref: https://github.com/mozilla/DeepSpeech/blob/master/bin/import_librivox.py**

```
wav_root - speaker_id - speaker_id-001.wav
                      - speaker_id-002.wav
                      - ...
```

+ Run preprocess.py


```python
python preprocess.py --in_dir /home/ninas96211/data/libri --pk_dir /home/ninas96211/data/libri_pickle --data_type libri
python preprocess.py --in_dir /home/ninas96211/data/vox1 --pk_dir /home/ninas96211/data/vox1_pickle --data_type vox1
python preprocess.py --in_dir /home/ninas96211/data/vox2 --pk_dir /home/ninas96211/data/vox2_pickle--data_type vox2
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

## Results

- Similarity Matrix

![alt text](https://github.com/Suhee05/Text-Independent-Speaker-Verification/blob/master/imgs/sim_mat.png?raw=true)

- Speaker Verification Task

After training 35000 steps using vox1 dataset, this model caught similarity between two waves from the same video clip, however in other cases, it was not successful. Currently this model using all 3 datasets(libri,vox1,vox2) is training and the result will be posted soon.


## Current Issues

- [@jaekukang](https://github.com/jaekookang) cloned this repository and he trained this model successfully. In inference.py, however, he found a bug. I fixed the bug.