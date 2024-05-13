import streamlit as st
import os
import re

# Define the directory containing the videos
video_directory = './samples/samples'

# Function to extract all video files and organize them by name
def list_videos(directory):
    videos = {}
    for file in os.listdir(directory):
        match = re.match(r'(\d+)_(.*?)_epoch(\d+)-global_step(\d+)_0\.mp4', file)
        if match:
            exp = int(match.group(1))
            name = match.group(1) + " " + match.group(2)
            epoch = int(match.group(3))
            step = int(match.group(4))
            if name not in videos:
                videos[name] = []
            videos[name].append((exp, epoch, step, file))
    
    # Sort videos by epoch and step for each name
    for name in videos:
        videos[name].sort(key=lambda x: (x[0], x[1], x[2]))
    return videos

# Load videos and prepare selection options
videos = list_videos(video_directory)
names = sorted(list(videos.keys()))

st.title('Video Viewer')

# Dropdown to select video series, with key to refresh on change
selected_name = st.selectbox('Select Video Series', names, key="select_series")

# Clear videos when the group name changes
if 'last_selected' not in st.session_state or st.session_state['last_selected'] != selected_name:
    for key in list(st.session_state.keys()):
        if key.startswith('video'):
            del st.session_state[key]
    st.session_state['last_selected'] = selected_name

# Display text file content if exists
text_file_path = os.path.join(video_directory, f"{selected_name}.txt")
if os.path.exists(text_file_path):
    with open(text_file_path, "r") as file:
        text_content = file.read()
    st.text_area("Description", text_content, height=150)

if selected_name:
    if st.button('Load All Videos'):
        for index in range(len(videos[selected_name])):
            video_path = os.path.join(video_directory, videos[selected_name][index][-1])
            st.session_state[f'video{index}'] = video_path
    
    # Display videos in an expandable section
    with st.expander(f"Available Videos for {selected_name}", expanded=True):
        for index, (exp, epoch, step, _) in enumerate(videos[selected_name]):
            video_label = f"Exp: {exp} - Epoch {epoch} - Step {step}"
            col1, col2 = st.columns([1, 3])
            with col1:
                if f'video{index}' not in st.session_state:
                    if st.button(f'Load Video: {video_label}', key=f'btn{index}'):
                        video_path = os.path.join(video_directory, videos[selected_name][index][-1])
                        st.session_state[f'video{index}'] = video_path
            with col2:
                if f'video{index}' in st.session_state:
                    st.markdown(f"**{video_label}**")
                    st.video(st.session_state[f'video{index}'])
