#!/bin/bash

#declare -i j=43
END=98
for ((j=51;j<=END;j++)); do


    echo "Starting server"
    python3 runserver.py --seed=${j} &
    #python3 runserver_fedadagrad.py --seed=${j} &
    sleep 5  # Sleep for 3s to give the server enough time to start


    for i in 2 3 6 8 ; do
        echo "Starting client $i"
        python3 runclient.py --hospital=${i} --seed=${j} &
        sleep 5
    done
    sleep 110
    echo "Start Local Training"


    sleep 5
    echo "Start client 3"

    python3 runlocal.py --hospital=3 --seed=${j}

    sleep 5
    echo "Start client 6"
    python3 runlocal.py --hospital=6 --seed=${j}

    sleep 5
    echo "Start client 8"
    python3 runlocal.py --hospital=8 --seed=${j}

    sleep 5
    echo "Start client 2" 
    python3 runlocal.py --hospital=2 --seed=${j}

    sleep 5
    #python3 rundraw.py
done
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
