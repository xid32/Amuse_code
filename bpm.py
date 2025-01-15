import os
import pandas as pd
import librosa
import tqdm

# Function to calculate average BPM for each interval
def calculate_bpm_intervals(y, sr, interval_duration=3):
    # avg bpm
    avg_bpm, _ = librosa.beat.beat_track(y, sr=sr)
    # Calculate total number of intervals
    min_interval = interval_duration - 0.5
    num_intervals = 20
    # Initialize list to store BPM values for each interval
    bpm_intervals = []
    # Divide the audio into intervals and calculate BPM for each interval
    for i in range(num_intervals):
        start_time = i * interval_duration
        end_time = (i + 1) * interval_duration
        # Extract audio segment for the current interval
        if end_time == 60:
            y_interval = y[start_time * sr:]
        else:
            y_interval = y[start_time * sr: end_time * sr]
        if len(y_interval) <= min_interval * sr:
            bpm_intervals.append(0)
            continue
        # Perform beat detection for the current interval
        tempo, _ = librosa.beat.beat_track(y_interval, sr=sr)
        # If no BPM detected, set it to 0
        if not tempo:
            bpm_intervals.append(0)
        else:
            bpm_intervals.append(tempo)
    return bpm_intervals, avg_bpm


# Function to calculate differential BPM for each interval
def diff_bpm(bpm_list, avg_bpm, rate=0.3):
    length = len(bpm_list)
    diff_list = [0]
    for b_i in range(1, length):
        m_bpm = bpm_list[b_i - 1]
        n_bpm = bpm_list[b_i]
        diff = abs(n_bpm - m_bpm)
        if diff >= avg_bpm * rate:
            diff_list.append(1)
        else:
            diff_list.append(0)
    return diff_list


# Process audio files in batches
def process_audio_batch(audio_folder, output_folder):
    # Get list of audio files in the folder
    audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.wav')]
    # Initialize DataFrame to store BPM values for each audio
    df = pd.DataFrame(columns=['Audio'] + [f'Interval_{i+1}' for i in range(20)])
    df_diff = pd.DataFrame(columns=['Audio'] + [f'Interval_{i+1}' for i in range(20)])
    # Process each audio file
    for audio_path in tqdm.tqdm(audio_files, total=len(audio_files)):
        # Read audio file
        y, sr = librosa.load(audio_path, sr=32000)
        # Calculate BPM values for each interval
        bpm_intervals, avg_bpm = calculate_bpm_intervals(y, sr)
        # diff
        diff = diff_bpm(bpm_intervals, avg_bpm, rate=0.3)
        # Add audio name and BPM values to DataFrame
        audio_name = os.path.basename(audio_path)
        df.loc[len(df)] = [audio_name] + bpm_intervals
        df_diff.loc[len(df)] = [audio_name] + diff
    # Save DataFrame as CSV file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'bpm_intervals.csv')
    diff_output_path = os.path.join(output_folder, 'diff_bpm_intervals.csv')
    df.to_csv(output_path, index=False)
    df_diff.to_csv(diff_output_path, index=False)
    return

# Main function
def main():
    # Set paths for input audio folder and output folder
    input_audio_folder = '/home/src/DDDD/wk/iwla_acc/datasets/audios'
    output_folder = '/home/src/DDDD/wk/iwla_acc/datasets/csv'
    # Process audio files in batches
    process_audio_batch(input_audio_folder, output_folder)
    return

if __name__ == "__main__":
    main()
