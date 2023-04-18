#!/bin/bash

usage() {
	echo "Usage:  $0 <directory>"
	exit 1
}

if [ $# -ne 1 ]; then
	usage
fi

directory="$1"

dirsize=$(ls -1 "$directory" | wc -l)

while true; do
	files=($(ls -1 "$directory" | while read line; do echo "$line"; done))
	no_files=${#files[@]}
	if [ "$no_files" -eq 0 ]; then
		echo "WAITING FOR FILE TO EXECUTE ..."
	elif [ "$no_files" -eq 1 ]; then
		script="$directory/${files[0]}"
		echo "EXECUTING FILE \"$script\" ..."
		sudo chmod +x "$script"
		"$script"
		echo "REMOVING FILE \"$script\" ..."
		sudo rm "$script"
		echo
	else
		echo "WARNING:  DIRECTORY \"$directory\" HAS MORE THAN ONE FILE IN IT. WILL EXECUTING NOTHING UNTIL THIS IS RESOLVED ..." 
	fi
	sleep 2
done
