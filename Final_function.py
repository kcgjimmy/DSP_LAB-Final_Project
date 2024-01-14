import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write as audio_write

def plot_spectrogram(y, sr, hop_length=512, n_fft=2048):

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft)), ref=np.max)
    

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()



def extract_top_n_frequencies(y, sr, hop_length=512, n_fft=2048, top_n=5):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft)), ref=np.max)
    top_n_frequencies = []


    for i in range(D.shape[1]):
        spectrum = D[:, i]
        top_n_indices = np.argsort(spectrum)[-top_n:]
        top_n_values = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[top_n_indices]
        
        top_n_frequencies.append(top_n_values)


    return np.array(top_n_frequencies)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))




def print_state(message, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    print(f"{colors[color]}{message}{colors['end']}")


def loading_STFT_File(database_path, music_paths, target_list, verbose=False):
    for i, path in enumerate(music_paths):
        music_name = os.path.splitext(os.path.basename(path))[0]
        npy_file_path = f'{database_path}{music_name}_STFT.npy'

        if os.path.exists(npy_file_path):
            # 如果 .npy 文件存在，直接加載
            top_n = np.load(npy_file_path)
            if(verbose): print(f'已加载 {npy_file_path}。')
        else:
            # 如果 .npy 文件不存在，進行STFT
            y, sr = librosa.load(path, sr=44100) ## 透過librosa讀取音樂
            top_n = extract_top_n_frequencies(y, sr) ## STFT
            np.save(npy_file_path, top_n)  ## 儲存資料到資料庫
            if(verbose): print(f'{npy_file_path} 不存在，已進行 STFT 並保存。')
            
        target_list.append(top_n)
        
def record_audio(duration, filename = "./record/output.wav", samplerate=44100, gain=3.0):
    # 錄製音訊
    os.makedirs('./record', exist_ok=True)
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype=np.int16)
    print("音檔錄製中...")
    print("錄製時間為: " + str(duration) + ' 秒。')
    # 等待音訊完成
    sd.wait()

    print("音檔錄製完成。")

    # 挑整大小聲
    audio_data = audio_data.astype(np.float64) 
    audio_data *= gain
    audio_data = audio_data.astype(np.int16) 
    audio_write(filename, samplerate, audio_data)
    return filename



def split_wav(file_path, output_folder, num_parts=20):
    # 讀取音檔
    y, sr = librosa.load(file_path, sr=None)

    # 計算每份sample數量
    samples_per_part = len(y) // num_parts

    # 解剖並儲存
    for i in range(num_parts):
        start_sample = i * samples_per_part
        end_sample = (i + 1) * samples_per_part

        partial_audio = y[start_sample:end_sample]
        partial_filename = os.path.join(output_folder, f'partial_{i + 1}.wav')
        sf.write(partial_filename, partial_audio, sr)