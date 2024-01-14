import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from Final_function import plot_spectrogram, extract_top_n_frequencies, euclidean_distance, print_state, loading_STFT_File

#TODO
target = './slices_Cover/song_4/partial_3.wav'
Libray = './Music/'
database_path = './Database/'


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
loading_STFT_File(database_path, music_paths, top_n_frequencies, verbose=True)


    
print('--------------------')
print_state('讀取目標音訊, 做STFT', 'green')
print('--------------------')
y_target, sr_target = librosa.load(target, sr=44100)
top_n_frequencies_target = extract_top_n_frequencies(y_target, sr_target)

# x = [i for i in range(len(top_n_frequencies_target))]
# for i in range(len(top_n_frequencies_target[0])):
#     plt.scatter(x, np.log10(top_n_frequencies_target[:, i]), c='r', s=1)
# plt.show()


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






# #     distances_matrix2.append([euclidean_distance(window_matrix1, top_n_frequencies0[j:j+window_size]) for j in range(len(top_n_frequencies0) - window_size + 1)]))
# #     # 計算與矩陣3的距離
# #     distances_matrix3.append(min([euclidean_distance(window_matrix1, top_n_frequencies1[k:k+window_size]) for k in range(len(top_n_frequencies1) - window_size + 1)]))
# # # 打印結果
# # print("最小距離矩陣2:", min(distances_matrix2))
# # print("最小距離矩陣3:", min(distances_matrix3))
# # 打印结果（每个窗口最大的10个频率）
# #print(len(top_n_frequencies))
# #print(top_n_frequencies)

# #plot_spectrogram(y, sr)
# # 86 is seconde
# # print(top_n_frequencies[0])
# # x=[i for i in range(len(top_n_frequencies))]
# # x1=[i for i in range(len(top_n_frequencies2))]
# # x2=[i for i in range(len(top_n_frequencies3))]
# # for i in range(len(top_n_frequencies[0])):
# #     #plt.scatter(x,top_n_frequencies[:,i],c='b')
# #     plt.scatter(x,np.log10(top_n_frequencies[:,i]),c='r',  s=1)
# #     plt.scatter(x1,np.log10(top_n_frequencies2[:,i]),c='b', s=1)
# #     plt.scatter(x2,np.log10(top_n_frequencies3[:,i]),c='g', s=1)

# # plt.show()


