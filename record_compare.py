import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from Final_function import plot_spectrogram, extract_top_n_frequencies, euclidean_distance, print_state, loading_STFT_File, record_audio

record_duration = 15
Libray = './Music/'
database_path = './Database/'

## 錄製音訊
target = record_audio(record_duration)


## 讀取音樂資料庫
music_files = [file for file in os.listdir(Libray) if file.endswith(('.wav', '.mp3'))]
music_names_list = []
for file in music_files:
    music_name = os.path.splitext(file)[0]
    music_names_list.append(music_name)
music_paths = [os.path.join(Libray, file) for file in music_files]
music_number = len(music_paths)






top_n_frequencies = []
## 檢查資料庫路徑是否存在，否則創建一個新的
if not os.path.exists(database_path):
    os.makedirs(database_path)

## 加載STFT(Numpy格式)，若無則創建並加載
print('--------------------')
print_state('加載資料庫中STFT檔案', 'green')
print('--------------------')
loading_STFT_File(database_path, music_paths, top_n_frequencies, verbose=False)


    
print('--------------------')
print_state('讀取目標音訊, 做STFT', 'green')
print('--------------------')
y_target, sr_target = librosa.load(target, sr=44100)
top_n_frequencies_target = extract_top_n_frequencies(y_target, sr_target)


## Tracing 開始
window_size = len(top_n_frequencies_target)
distance_list = []
for i in range(len(top_n_frequencies_target) - window_size + 1):
    for k in range(len(top_n_frequencies)):
        distance = [euclidean_distance(top_n_frequencies_target, top_n_frequencies[k][j:j+window_size]) for j in range(len(top_n_frequencies[k]) - window_size + 1)]
        distance_list.append(distance)
chooser = []
for i in range(len(distance_list)): chooser.append(min(distance_list[i]))



## 顯示預估結果
print('--------------------')
print_state('    最終判斷結果', 'red')
print('--------------------')
min_index = np.argmin(chooser)
min_result = os.path.splitext(music_files[min_index])[0]
print("有可能的結果: " + min_result)


