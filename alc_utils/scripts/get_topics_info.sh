#!/bin/bash
# Script for getting a list of all ROS topics with detailed descriptions.
# First runs "rostopic list" to get a complete list of published topics
# Then runs "rostopic info *" for each topic in generated list to get detailed info.

# Desired file names
names_file="topic_names.txt"
details_file="topic_details.txt"

# Get complete list of all currently-published topics
rostopic list > ${names_file}

# Overwrite any existing details file
echo "" > ${details_file}

# Call 'rostopic info' on each topic and store results
while IFS= read -r line
do
  echo "Topic: ${line}" >> ${details_file}
  rostopic info ${line} >> ${details_file}
  echo "#############################################" >> ${details_file}
done < "${names_file}"


# TODO: Make this collect info about ROS Services as well