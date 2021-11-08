#!/usr/bin/env bash

DOCKER_NAME="objectnav_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -v /home/ubuntu/habitat_data/challenge-data:/habitat-challenge-data \
    -v /home/ubuntu/habitat_data:/habitat-challenge-data/data \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/configs/challenge_objectnav2021.local.rgbd.yaml" \
    ${DOCKER_NAME}\
