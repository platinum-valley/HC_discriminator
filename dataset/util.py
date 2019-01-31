import re
import os
import wave
import pandas as pd
import numpy as np
import pickle
from wav2mfcc import wav_to_mfcc

def make_feature(wav, offset, duration):
    feature = wav_to_mfcc(wav, offset, duration)
    return feature

def rename_file(dir_path):

    files = os.listdir(dir_path)
    for (i,file_name) in enumerate(files):
        os.rename("{}/{}".format(dir_path, file_name), "{}/{}.wav".format(dir_path,str(i+1).zfill(4)))

def get_music_duration(wav_file):
    wf = wave.open(wav_file, "r")
    return (float(wf.getnframes()) / wf.getframerate())

def get_music_offset(duration, delta_duration):
    delta_duration = 10.0
    default_offset = 30.0

    offset_list = []
    offset = default_offset
    while True:
        if duration > offset + delta_duration:
            offset_list.append(offset)
            offset += delta_duration
        else:
            break
    return offset_list

def make_dataset_csv(dir_path_list, label_list,output_file_path):
    delta_duration = 10.0
    dataset_list = []
    csv_dataset_list = []
    feat_count = 1
    for (i, dir_path) in enumerate(dir_path_list):
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            file_path = dir_path + "/" + file_name
            duration = get_music_duration(file_path)
            offset_list = get_music_offset(duration, delta_duration)
            for offset in offset_list:
                pkl_path = "feature/{}.pkl".format(str(feat_count).zfill(6))
                feat_count += 1
                with open(pkl_path, "wb") as f:
                    pickle.dump(make_feature(file_path, offset, delta_duration), f)
                    csv_dataset_list.append([pkl_path, label_list[i]])
                dataset_list.append([make_feature(file_path, offset, delta_duration), label_list[i]])
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset_list, f)
    dataframe = pd.DataFrame(csv_dataset_list)
    dataframe.to_csv(output_file_path, header=False, index=False)

if __name__ == "__main__":
    #rename_file("music_dataset/hardcore")
    #rename_file("music_dataset/not_hardcore")
    make_dataset_csv(["music_dataset/hardcore", "music_dataset/not_hardcore"], ["True", "False"],  "dataset.csv")
