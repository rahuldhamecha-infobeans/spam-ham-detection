import subprocess
import re
import cv2
import os
from datetime import datetime
current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
from fer import FER
import json
import glob
emotion_detector = FER(mtcnn=True)
import math

def count_images_in_directory(directory_path, extensions=("*.jpg", "*.jpeg", "*.png", "*.gif")):
    # Create a list comprehension to generate a list of image files for each extension
    image_files = [file for ext in extensions for file in glob.glob(os.path.join(directory_path, ext))]
    
    # Calculate the total count of image files
    image_count = len(image_files)

    return image_count

# Define the timestamp_to_seconds function
def timestamp_to_seconds(timestamp):
    components = timestamp.split(':')
    if len(components) == 2:
        # If only minutes and seconds are provided, assume hours as 0
        hours, minutes, seconds = 0, float(components[0]), float(components[1])
    elif len(components) == 3:
        hours, minutes, seconds = map(float, components)
    else:
        raise ValueError("Invalid timestamp format")

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def generate_transcipt(videopath):
    # Run the Whisper command and capture its output
    command = f"whisper '{videopath}' --model tiny --language English"  # Use videopath variable
    output = subprocess.check_output(command, shell=True, text=True)

    # Initialize a list to store the timestamped transcript lines
    timestamps = []

    # Initialize variables to track the current speaker
    current_speaker = None

    # Process the Whisper output and extract timestamped transcript lines
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line:  # Check if the line is not empty
            # Extract timestamps and transcript text using regular expression
            match = re.match(r'\[(\d+:\d+\.\d+) --> (\d+:\d+\.\d+)\] (.+)', line)
            if match:
                start_time, end_time, transcript = match.groups()
            
                # Determine the speaker based on the presence of a "?" mark
                if "?" in transcript:
                    current_speaker = "Interviewer"
                else:
                    current_speaker = "candidate"
            
                timestamps.append({
                    'speaker': current_speaker,
                    'start': timestamp_to_seconds(start_time),
                    'end': timestamp_to_seconds(end_time),
                    'transcript': transcript
                })

    merged_data = []
    current_entry = None

    for entry in timestamps:
        if current_entry is None:
            current_entry = entry
        elif current_entry['speaker'] == entry['speaker']:
            current_entry['end'] = entry['end']
            current_entry['transcript'] += entry['transcript']
        else:
            merged_data.append(current_entry)
            current_entry = entry

    if current_entry is not None:
        merged_data.append(current_entry)

    return(merged_data)



def save_frames_for_timestamps(video_path, timestamps, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    os.makedirs(dir_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_inv = 1 / fps
    
    for i, timestamp in enumerate(timestamps):
        if timestamp.start_duration ==0:
            start_time_sec = math.ceil(float(timestamp.start_duration))+1
        else:
            start_time_sec = math.ceil(float(timestamp.start_duration))
        end_time_sec = math.ceil(float(timestamp.end_duration))
        screenshot_count = 0
        current_time = start_time_sec
        frame_time_interval = 1.0 
        sub_dir_path = os.path.join(dir_path, f'{timestamp.id}__timestamp_{start_time_sec}_{end_time_sec}')
        os.makedirs(sub_dir_path, exist_ok=True)

        while current_time < end_time_sec:
            n = round(fps * current_time)
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(
                    os.path.join(sub_dir_path, '{}_{}.{}'.format(basename, screenshot_count, ext)),
                    frame
                )
                screenshot_count += 1
            else:
                break
            current_time += frame_time_interval  # Capture one frame per second

    # Release the video capture
    cap.release()
    return True

# Usage
#save_frames_for_timestamps('interview2.mp4', timestamps, f'videoframes/{current_datetime}', 'interview_frame')


def analyze_timestamp_folder(timestamp_folder):
    # Initialize counters for each emotion for emotion_1 and emotion_2

    final_json_result=[]
    # Iterate over all timestamp folders
    for timestamp_dir in os.listdir(timestamp_folder):
        timestamp_dir_path = os.path.join(timestamp_folder, timestamp_dir)

        if os.path.isdir(timestamp_dir_path):  # Check if it's a directory
            # Initialize an empty list to store the analysis results for each image in the timestamp folder
            analysis_results = []
            emotion_totals_1 = {
                "angry": 0,
                "disgust": 0,
                "fear": 0,
                "happy": 0,
                "sad": 0,
                "surprise": 0,
                "neutral": 0
            }

            emotion_totals_2 = {
                "angry": 0,
                "disgust": 0,
                "fear": 0,
                "happy": 0,
                "sad": 0,
                "surprise": 0,
                "neutral": 0
            }
            print(timestamp_dir_path)
            count = count_images_in_directory(f'{timestamp_dir_path}/')

            # Iterate over all image files in the current timestamp folder
            for filename in os.listdir(timestamp_dir_path):
                if filename.endswith(".jpg"):  # Ensure you are processing only image files
                    image_path = os.path.join(timestamp_dir_path, filename)
                    test_img = cv2.imread(image_path)

                    # Detect emotions in the current image
                    analysis = emotion_detector.detect_emotions(test_img)

                    # Append the analysis result to the list
                    analysis_results.append(analysis)
                    analysis_results_len =   len(analysis)
                    
                    if len(analysis)!=0 and len(analysis) < 2:
                        emotion_1 = analysis[0]['emotions']
                        emotion_totals_1["angry"] += emotion_1['angry'] 
                        emotion_totals_1["disgust"] += emotion_1['disgust'] 
                        emotion_totals_1["fear"] += emotion_1['fear'] 
                        emotion_totals_1["happy"] += emotion_1['happy'] 
                        emotion_totals_1["sad"] += emotion_1['sad'] 
                        emotion_totals_1["surprise"] += emotion_1['surprise'] 
                        emotion_totals_1["neutral"] += emotion_1['neutral'] 
                    elif len(analysis)<3 and len(analysis) > 1 :
                        # Extract emotions for emotion_1 and emotion_2
                        emotion_1 = analysis[0]['emotions']
                        emotion_2 = analysis[1]['emotions']

                        # Update emotion totals for emotion_1
                        emotion_totals_1["angry"] += emotion_1['angry'] 
                        emotion_totals_1["disgust"] += emotion_1['disgust'] 
                        emotion_totals_1["fear"] += emotion_1['fear'] 
                        emotion_totals_1["happy"] += emotion_1['happy'] 
                        emotion_totals_1["sad"] += emotion_1['sad'] 
                        emotion_totals_1["surprise"] += emotion_1['surprise'] 
                        emotion_totals_1["neutral"] += emotion_1['neutral'] 

                        # Update emotion totals for emotion_2
                        emotion_totals_2["angry"] += emotion_2['angry'] 
                        emotion_totals_2["disgust"] += emotion_2['disgust'] 
                        emotion_totals_2["fear"] += emotion_2['fear'] 
                        emotion_totals_2["happy"] += emotion_2['happy'] 
                        emotion_totals_2["sad"] += emotion_2['sad'] 
                        emotion_totals_2["surprise"] += emotion_2['surprise'] 
                        emotion_totals_2["neutral"] += emotion_2['neutral'] 
            
            # Create JSON objects for emotion totals
            timestamp_string = timestamp_dir
            parts = timestamp_string.split("__")
            emotion_totals_1_all_zero = all(value == 0.0 for value in emotion_totals_1.values())
            if emotion_totals_1_all_zero:
                result_1={}
            else:
                result_1 = {key: round(value / count, 2) for key, value in emotion_totals_1.items()}
            emotion_totals_2_all_zero = all(value == 0.0 for value in emotion_totals_2.values())
            if emotion_totals_2_all_zero:
                result_2={}
            else:
                result_2 = {key: round(value / count, 2) for key, value in emotion_totals_2.items()}

            if  result_1 and result_2:
                if result_1["neutral"] > result_2["neutral"] and result_1["happy"] > result_2["happy"]:
                    selected_json = result_1
                else:
                    selected_json = result_2
                json_result = {parts[0]: selected_json}
            elif result_1 and not result_2:
                json_result = {parts[0]: result_1}
               

            final_json_result.append(json_result) 
        
    return final_json_result