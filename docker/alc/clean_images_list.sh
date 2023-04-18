#!/bin/bash

image_list=("alc:latest"
            "alc_dind:latest"
            "xpra_16.04:latest"
            )

server_name="git.isis.vanderbilt.edu:5050/alc/alc"

for (( i=0; i<${#image_list[*]}; ++i)); do
    image_name=${image_list[$i]}
    image_tagged_name=${server_name}/$image_name
    docker rmi $image_tagged_name
done
