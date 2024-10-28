#!/bin/bash

#VIDEO_FOLDER="/home/manu/mnt/ST8000DM004-2U91/smoke/data/test/烟雾/正例（200）"
#VIDEO_FOLDER="/home/manu/mnt/ST8000DM004-2U91/smoke/data/test/烟雾/反例（200）"
#VIDEO_FOLDER="/media/manu/ST2000DM005-2U91/fire/data/test/V3/positive"
VIDEO_FOLDER="/media/manu/ST2000DM005-2U91/fire/data/test/V3/negative"
#VIDEO_FOLDER="/media/manu/ST2000DM005-2U91/fire/data/test/V3"
PYTHON_SCRIPT="demos/demo_detect_mp.py"
SAVE_ROOT="/home/manu/tmp/fire_test_results"

rm -rf $SAVE_ROOT
mkdir -p $SAVE_ROOT

find "$VIDEO_FOLDER" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | while read -r video_file; do
    echo "Processing video: $video_file"
    python "$PYTHON_SCRIPT" --path_video "$video_file" --save_root "$SAVE_ROOT"
    if [ $? -ne 0 ]; then
        echo "Error processing video: $video_file"
    else
        echo "Finished processing video: $video_file"
    fi
done
