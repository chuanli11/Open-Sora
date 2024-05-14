#!/bin/bash

SESSION="nvtop_monitor"
NODE_FILE=$1

# Check if the node file exists
if [ ! -f $NODE_FILE ]; then
    echo "Node file $NODE_FILE not found!"
    exit 1
fi

# Create a new tmux session
tmux new-session -d -s $SESSION

# Read nodes from the file and create panes
i=0
while IFS= read -r NODE; do
    if [ $i -ne 0 ]; then
        tmux split-window -t $SESSION
        tmux select-layout -t $SESSION tiled
    fi
    tmux send-keys -t $SESSION "ssh -t $NODE 'nvtop'" C-m
    i=$((i + 1))
done < "$NODE_FILE"

# Attach to the tmux session
tmux attach -t $SESSION

