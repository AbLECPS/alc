#!/bin/bash

DIRECTORY="/alc/workflows/config"
WEBGME_BUILD_HOME="/alc/webgme/automate/gradle"

mkdir -p "$DIRECTORY"

while true; do
    readarray -t files < <(ls -1 "$DIRECTORY")
    if [ ${#files[*]} -eq 0 ]; then
      sleep 1
    else
      file="$DIRECTORY/${files[0]}"
      workflowDir="$(cat "$file" | jq -r ".buildRoot")"
      runDir="$workflowDir/run/main"
      mkdir -p "$runDir"
      "$WEBGME_BUILD_HOME/gradlew" iterate -b "$WEBGME_BUILD_HOME/build.gradle.kts" -Pconfig_file="$file" \
         > "$runDir/build.stdout" 2> "$runDir/build.stderr"
      rm -f "$file"
    fi
done
