#!/bin/bash
REPO_DIR="."
TRAIN_SCRIPT="scripts/auto_train.py"
LAST_COMMIT_FILE="/tmp/last_commit.txt"

cd $REPO_DIR

# Fetch latest changes
git fetch
git pull

# Get latest commit hash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
LATEST_COMMIT_HASH=$(git rev-parse $CURRENT_BRANCH)
LAST_COMMIT_HASH=$(cat $LAST_COMMIT_FILE 2>/dev/null)

# Check if new commit exists
if [ "$LATEST_COMMIT_HASH" != "$LAST_COMMIT_HASH" ]; then
    LATEST_COMMIT_MSG=$(git log -1 --pretty=%B $CURRENT_BRANCH)

    if [[ "$LATEST_COMMIT_MSG" == *"[TRAIN]"* ]]; then
        echo "New [TRAIN] commit found. Running training..."
        nohup /usr/bin/python3 -m uv run $TRAIN_SCRIPT > train.log 2>&1 &
    fi
    # Update last processed commit
    echo $LATEST_COMMIT_HASH > $LAST_COMMIT_FILE
fi

