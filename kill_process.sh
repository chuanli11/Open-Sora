#!/bin/bash

# Check if the input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi


input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Error: Input file '$input_file' does not exist."
  exit 1
fi

while IFS= read -r hostname || [ -n "$hostname" ]; do
  if [ -n "$hostname" ]; then
    echo "Checking active Conda environment and killing processes using its Python on $hostname"
    active_env=$CONDA_DEFAULT_ENV
    ssh "$hostname" "
      if [ -n \"$active_env\" ]; then
        for pid in \$(pgrep -f \"$active_env/bin/python\"); do
          echo \"Killing process \$pid using the active Conda environment Python $active_env.\"
          sudo kill -9 \$pid
        done
      else
        echo \"No active Conda environment found\"
      fi" &
  fi
done < "$input_file"

