from Final_function import split_wav
import os


## 讀取音樂切片路徑
input_path = './Music/'
output_path = './slices_Music/'
post_fix = '.wav'
input_names = os.listdir(input_path)
input_paths = [os.path.join(input_path, file) for file in input_names]
prefix_list = [item.split(post_fix)[0] for item in input_names]





for i in range(len(prefix_list)):
    input_file_path = input_paths[i]                        # Input .wav file path
    output_folder = output_path + prefix_list[i]             # Folder to save partial audio
    # Create the folder to save partial audio
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split audio and save
    split_wav(input_file_path , output_folder, num_parts=20)
