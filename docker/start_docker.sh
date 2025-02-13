#!/bin/bash
export UID
export USER

docker_list=$(docker ps --filter "name=odp_dev" -q)

if [ -z "$docker_list" ]; then
        echo "We have no running docker container, so we start one."
        docker compose -p b2h -f ./docker-compose.yml run -d --name odp_dev --rm --service-ports odp_dev
else
        echo "We are entering the only running docker container: $docker_list"
        docker exec -it -e HOST_USER=$USER -e HOST_UID=$UID -u 0 $docker_list /user_entry.sh
fi