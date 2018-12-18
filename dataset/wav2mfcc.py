# coding: utf-8

import librosa
import re
import numpy as np
import scipy
import os
from pydub import AudioSegment


def mp3_to_wav(mp3_file_path):
    """
    mp3形式の音声ファイルをwav形式の音声ファイルに変換する

    :param mp3_file_path: mp3形式の音声ファイルのパス
    """
    wav = AudioSegment.from_mp3(mp3_file_path)
    return wav

def wav_to_mfcc(wav, off, dur):
    """
    wav形式の音声ファイルからmfcc(メル周波数ケプストラム係数)を取得する

    :type file_path: wav形式の音声ファイルのパス
    :return mfcc: 音声ファイルより得られるmfcc
    """
    x, fs = librosa.load(wav, sr=44100, offset=off, duration=dur)
    mfcc = librosa.feature.mfcc(x, sr=fs)
    return mfcc


def mean_vector(vector):  # 平均
    return np.mean(vector)


def max_vector(vector):   # 最大値
    return np.max(vector)


def min_vector(vector):   # 最小値
    return np.min(vector)


def var_vector(vector):   # 分散
    return np.var(vector)


def sum_vector(vector):   # 合計
    return np.sum(vector)


def med_vector(vector):   # 中央値
    return np.median(vector)


def kurtosis_vector(vector):  # 尖度
    return scipy.stats.kurtosis(vector)


def skew_vector(vector):   # 歪度
    return scipy.stats.skew(vector)


def nobs_vector(vector):   # 要素数(時間)
    return vector.shape[0]


def calculate_stat(matrix, sequence_length):
    """
    mfccの特徴量ごとに統計量を計算する
    statに示す統計量を特徴量とする
    (data_num, sequence_length ,mffc_dimension=10 * 統計量の数)の特徴量を抽出する

    :param matrix: mfcc行列
    :param sequence_length: 系列長
    :return: 特徴量ベクトル
    """
    stat = ["mean", "max", "min", "var", "sum", "med", "kurtosis", "skew"]
    stat_matrix = np.zeros((sequence_length, matrix.shape[0], len(stat)))
    print(matrix.shape)
    for i in range(sequence_length):
        for j in range(matrix.shape[0]):
            divide_sequence = int(matrix.shape[1] / sequence_length)
            for k in range(len(stat)):
                if i == sequence_length - 1:
                    stat_matrix[i][j][k] = eval(stat[k] + "_vector")(matrix[j][i*divide_sequence : (i+1)*divide_sequence + matrix.shape[1]%sequence_length])
                else:
                    stat_matrix[i][j][k] = eval(stat[k] + "_vector")(matrix[j][i*divide_sequence : (i+1)*divide_sequence])
    sequence_vec = np.reshape(stat_matrix, (sequence_length, -1))
    print(sequence_vec.shape)
    return sequence_vec


def mp3_to_vector(mp3_directory_path, mp3_file_name):
    """
    mp3形式の音声ファイルからmfccの特徴量ベクトルを抽出する

    :param mp3_directory_path: mp3_file_nameのあるカレントディレクトリを指す
    :param mp3_file_name: 対象とするmp3形式の音声ファイル
    """
    os.chdir(mp3_directory_path)
    mp3_to_wav(mp3_file_name)
    wav_file_name = re.sub("mp3", "wav", mp3_file_name)
    mfcc_vector = calculate_stat(wav_to_mfcc(wav_file_name), 30)
    return mfcc_vector

def mp3_to_mfcc(mp3_directory_path, offset, duration):
    wav = mp3_to_wav(mp3_directory)
    mfcc = wav_to_mfcc(wav, offset, duration)
    return mfcc

if __name__ == "__main__":
    mp3_to_wav("./The Ripper.mp3")
    mfcc = wav_to_mfcc("The Ripper.wav")
    vec = calculate_stat(mfcc)
    print(vec.shape)
