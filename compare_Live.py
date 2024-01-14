import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from Final_function import plot_spectrogram, extract_top_n_frequencies, euclidean_distance, print_state, loading_STFT_File
from sklearn.metrics import confusion_matrix
import seaborn as sns


Libray = './Music/'
database_path = './Database/'

## 讀取音樂切片路徑
slices_path = './slices_Live'
slices_names = os.listdir(slices_path)
slices_paths = [os.path.join(slices_path, file) for file in slices_names]



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




# 創建混淆矩陣
truth_labels = []
predicted_labels = []


    

## 顯示預估結果
print('--------------------')
print_state('    最終判斷結果', 'red')
print('--------------------')

for slice_select_index in range(len(slices_names)):
    
    ## 選擇比較音樂資料夾
    slices_select = slices_paths[slice_select_index]
    slices_select_name = os.path.splitext(os.path.basename(slices_select))[0]
    slices_files = [file for file in os.listdir(slices_select) if file.endswith(('.wav'))]
    slices_select_paths = [os.path.join(slices_select, file) for file in slices_files]
    
    Truth = 0
    Wrong = 0
    
    ## 比較內部所有音檔
    for target in slices_select_paths:
        y_target, _ = librosa.load(target, sr=44100)
        top_n_frequencies_target = extract_top_n_frequencies(y_target, 44100)
        ## Tracing 開始
        window_size = len(top_n_frequencies_target)
        distance_list = []
        for i in range(len(top_n_frequencies_target) - window_size + 1):
            for k in range(len(top_n_frequencies)):
                distance = [euclidean_distance(top_n_frequencies_target, top_n_frequencies[k][j:j+window_size]) for j in range(len(top_n_frequencies[k]) - window_size + 1)]
                distance_list.append(distance)
        chooser = []
        for i in range(len(distance_list)): chooser.append(min(distance_list[i]))
        
        min_index = np.argmin(chooser)
        min_result = os.path.splitext(music_files[min_index])[0]
        truth_labels.append(slices_select_name)
        predicted_labels.append(min_result)
        
        print("有可能的結果: " + min_result)
        if(min_result == slices_select_name): 
            Truth = Truth+1
        else: 
            Wrong = Wrong+1

    print('總數:', Truth + Wrong)
    print('正確數:', Truth)
    print('錯誤數:', Wrong)






# 繪製混淆矩陣
conf_matrix = confusion_matrix(truth_labels, predicted_labels, labels=music_names_list)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=music_names_list, yticklabels=music_names_list, cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix')
plt.show()


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


