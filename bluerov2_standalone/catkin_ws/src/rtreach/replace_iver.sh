#/bin/bash
input="iver_strings.txt"
while read -r line
do
  echo "$line"
  sed -i 's/ng_msgs/vandy_bluerov/g' $line
done < "$input"