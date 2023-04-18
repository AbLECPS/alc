#!/bin/bash

# bag2csv -- Script to extract all topics from a ROS recording bag file and output them to a csv file

if [ "$1" == "" ]; then
	echo "No bag file name specified"
	exit
else
	# Remove file type extension from filename if present
	temp=$1
	filename=${temp%.*}
fi

for topic in `rostopic list -b $filename.bag`; do 
	if [[ $topic = *"image"* ]]; then
		echo "WARNING: Skipping possible image data "$topic
	else
		echo "Reading topic "$topic"..."
		rostopic echo -p -b $filename.bag $topic > $filename${topic//\//_}.csv; 
	fi
done
