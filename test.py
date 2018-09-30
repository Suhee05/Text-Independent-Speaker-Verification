import tensorflow as tf
import numpy as np
import glob
import queue
import argparse
from model import GE2E
from feeder import Feeder



def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # Path

    # wav name formatting: id_clip_uttnum.wav
    parser.add_argument("--test_dir", type=str, required=True, help="input test dir")
    #/home/hdd2tb/ninas96211/dev_wav_set

    parser.add_argument("--ckpt_file", type=str, required=True, help="checkpoint to start with for inference")

    # Data
    #parser.add_argument("--window_length", type=int, default=160, help="sliding window length(frames)")
    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="overlaping percentage")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")
    # Enrol
    parser.add_argument("--num_spk_per_batch", type=int, default= 5,
                                           help="N speakers of batch size N*M")
    parser.add_argument("--num_utt_per_batch", type=int, default= 10,
                                           help="M utterances of batch size N*M")

    # LSTM
    parser.add_argument("--num_lstm_stacks", type=int, default=3, help="number of LSTM stacks")
    parser.add_argument("--num_lstm_cells", type=int, default=768, help="number of LSTM cells")
    parser.add_argument("--dim_lstm_projection", type=int, default=256, help="dimension of LSTM projection")


	# Total Score
	# (Sum of True Scores) -  (Sum of False Scores)
	# if the model is perfect, the score will be num_of_true_pairs
	# if the model really sucks, the score will be - num_of_false_pairs 

    # Collect hparams
    args = parser.parse_args()
    global_queue = queue.Queue()
    feeder = Feeder(args, "test")
    feeder.set_up_feeder(global_queue)

    model = GE2E(args)
    graph = model.set_up_model("test")




    with graph.as_default():
        saver = tf.train.Saver()
        
    with tf.Session(graph=graph) as sess:
        # restore from checkpoints

        saver.restore(sess, args.ckpt_file)
        total_score = 0
        num_true_pairs= 0
        num_false_pairs = 0

        while len(feeder.wav_pairs) > 0:
            wav1_data, wav2_data, match = global_queue.get()
            wav1_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav1_data})
            wav2_out = sess.run(model.norm_out, feed_dict={model.input_batch:wav2_data})
            wav1_dvector = np.mean(wav1_out, axis=0)
            wav2_dvector = np.mean(wav2_out, axis=0)
            final_score = np.dot(wav1_dvector, wav2_dvector)/(np.linalg.norm(wav1_dvector) * np.linalg.norm(wav2_dvector))
            print("final score:" + str(final_score))
            print("same? :" + str(match))
            if match == True:
                total_score += final_score
                num_true_pairs += 1
            if match == False:
                total_score -= final_score
                num_false_pairs += 1

    print("in total: " + str(total_score))
    print("num true pairs: " + str(num_true_pairs))
    print("num false pairs: " + str(num_false_pairs))


if __name__ == "__main__":
    main()

