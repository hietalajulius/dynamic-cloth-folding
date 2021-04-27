#!/bin/bash
set -e

# Creates a video from a folder containing a bunch of jpeg images.
##### Usage ####
# ./make_video.sh <path-to-folder>

pushd $1
ffmpeg -framerate 10 -pattern_type glob -i '*.png' \
    -c:v libx264 -pix_fmt yuv420p -y out.mp4
popd
