#!/bin/bash

# Script to create the next day folder based on existing day folders

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find all folders starting with "day" followed by numbers
# Extract the numbers and find the maximum
MAX_DAY=$(ls -d day[0-9]* 2>/dev/null | sed 's/day//' | sort -n | tail -1)

# If no day folders exist, start with day1
if [ -z "$MAX_DAY" ]; then
    NEXT_DAY=1
else
    # Increment the maximum day number
    NEXT_DAY=$((MAX_DAY + 1))
fi

# Create the new day folder with triton and cuda subfolders
NEW_FOLDER="day${NEXT_DAY}"
mkdir -p "$NEW_FOLDER/triton"
mkdir -p "$NEW_FOLDER/cuda"

echo "Created folder: $NEW_FOLDER with triton and cuda subfolders"

# Create git branch for the new day
BRANCH_NAME="day${NEXT_DAY}"
if git rev-parse --git-dir > /dev/null 2>&1; then
    git checkout -b "$BRANCH_NAME"
    echo "Created and switched to git branch: $BRANCH_NAME"
else
    echo "Warning: Not a git repository. Skipping branch creation."
fi

