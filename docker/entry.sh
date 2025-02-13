#!/usr/bin/env bash
if [ -z "$1" ]; then
    START_SSH=false
else
    START_SSH=$1
    shift
fi

if [ "$START_SSH" = true ]; then
    service ssh start
fi
tail -f /dev/null