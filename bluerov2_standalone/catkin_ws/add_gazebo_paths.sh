#!/bin/bash

find -name package.xml -print0 | xargs -0 -I? sh -c $'echo ?;
                                                      filename="$(echo ?)";
                                                      echo $filename;
                                                      prefix=$PWD"/${filename%%/package.xml}";
                                                      echo $prefix;
                                                      echo \'cat /package/export/gazebo_ros/@*[starts-with(name(),"gazebo")]\' | xmllint --shell ? >> temp.txt;
                                                      sed -i "s/ //g" temp.txt;
                                                      sed -i "s/\/>//g" temp.txt;
                                                      sed -i "s/-------//g" temp.txt;
                                                      sed -i \'s,\${prefix},\'"$prefix"\',g\' temp.txt;'

./add_gazebo_paths.py temp.txt > export_paths.sh
. export_paths.sh
rm temp.txt
rm export_paths.sh
