#!/bin/bash
REPO_DIR="."
TRAIN_SCRIPT="auto_train.py"
LAST_COMMIT_FILE="/tmp/last_commit.txt"

cd $REPO_DIR

# Fetch latest changes
git fetch origin main
git pull

# Get latest commit hash
LATEST_COMMIT_HASH=$(git rev-parse origin/main)
LAST_COMMIT_HASH=$(cat $LAST_COMMIT_FILE 2>/dev/null)

# Check if new commit exists
if [ "$LATEST_COMMIT_HASH" != "$LAST_COMMIT_HASH" ]; then
    LATEST_COMMIT_MSG=$(git log -1 --pretty=%B origin/main)

    if [[ "$LATEST_COMMIT_MSG" == *"[TRAIN]"* ]]; then
        echo "New [TRAIN] commit found. Running training..."
        git pull origin main
        nohup /home/pbenito/miniconda3/bin/python3 $TRAIN_SCRIPT > train.log 2>&1 &
    fi
    # Update last processed commit
    echo $LATEST_COMMIT_HASH > $LAST_COMMIT_FILE
fi

