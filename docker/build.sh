#!/bin/bash

docker build docker \
             -t bbq_image-demo \
             --build-arg UID=${UID} \
             --build-arg GID=${UID}