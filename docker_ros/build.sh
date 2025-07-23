#!/bin/bash

docker build docker_ros \
             -t bbq_image-ros1 \
             --build-arg UID=${UID} \
             --build-arg GID=${UID}