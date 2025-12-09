
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=rdp:v2
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        RDP-new \
        $BASH_OPTION
else
    docker start -i RDP-new
fi
