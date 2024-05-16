#!/bin/bash

# Check if the video folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/video/folder"
    exit 1
fi

# Define directories
ROOT_VIDEO="$1"
ROOT_CLIPS="${ROOT_VIDEO}/clips"
ROOT_META="${ROOT_VIDEO}/meta"
LOG_FILE="${ROOT_META}/process.log"

# Create directories if they do not exist
mkdir -p "${ROOT_CLIPS}"
mkdir -p "${ROOT_META}"

# Create or clear the log file
> "${LOG_FILE}"

# Function to run a command and log its output
run_command() {
    echo "Running: $1" | tee -a "${LOG_FILE}"
    eval "$1" >> "${LOG_FILE}" 2>&1
    if [ $? -ne 0 ]; then
        echo "Command failed: $1" | tee -a "${LOG_FILE}"
        exit 1
    fi
}

# 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
run_command "python -m tools.datasets.convert video \"${ROOT_VIDEO}\" --output \"${ROOT_META}/meta.csv\""

# 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin1.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta.csv\" --info --fmin 1"

# 2.1 Detect scenes. This should output ${ROOT_META}/meta_info_fmin1_timestamp.csv
run_command "python -m tools.scene_cut.scene_detect \"${ROOT_META}/meta_info_fmin1.csv\""

# 2.2 Cut video into clips based on scenes. This should produce video clips under ${ROOT_CLIPS}
run_command "python -m tools.scene_cut.cut \"${ROOT_META}/meta_info_fmin1_timestamp.csv\" --save_dir \"${ROOT_CLIPS}\""

# 2.3 Create a meta file for video clips. This should output ${ROOT_META}/meta_clips.csv
run_command "python -m tools.datasets.convert video \"${ROOT_CLIPS}\" --output \"${ROOT_META}/meta_clips.csv\""

# 2.4 Get clips information and remove broken ones. This should output ${ROOT_META}/meta_clips_info_fmin1.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta_clips.csv\" --info --fmin 1"

# 3.1 Predict aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_part*.csv
run_command "torchrun --nproc_per_node 8 -m tools.scoring.aesthetic.inference \"${ROOT_META}/meta_clips_info_fmin1.csv\" --bs 1024 --num_workers 16"

# 3.2 Merge files; This should output ${ROOT_META}/meta_clips_info_fmin1_aes.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta_clips_info_fmin1_aes_part*.csv\" --output \"${ROOT_META}/meta_clips_info_fmin1_aes.csv\""

# 3.3 Filter by aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta_clips_info_fmin1_aes.csv\" --aesmin 5"

# 4.1 Generate caption. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv
run_command "torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava \"${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.csv\" --dp-size 8 --tp-size 1 --model-path /path/to/llava-v1.6-mistral-7b --prompt video"

# 4.2 Merge caption results. This should output ${ROOT_META}/meta_clips_caption.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv\" --output \"${ROOT_META}/meta_clips_caption.csv\""

# 4.3 Clean caption. This should output ${ROOT_META}/meta_clips_caption_cleaned.csv
run_command "python -m tools.datasets.datautil \"${ROOT_META}/meta_clips_caption.csv\" --clean-caption --refine-llm-caption --remove-empty-caption --output \"${ROOT_META}/meta_clips_caption_cleaned.csv\""

echo "Video processing completed successfully!" | tee -a "${LOG_FILE}"
