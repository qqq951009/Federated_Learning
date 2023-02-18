#!/bin/bash

#declare -i j=42
seer=0
END=102
for ((j=42;j<=END;j++)); do

    python3 make_encode_map.py --seed=${j} --seer=${seer}
    
    echo "Starting seed" ${j}
    python3 runserver.py --seed=${j} --seer=${seer} &
    sleep 1  # Sleep for 3s to give the server enough time to start


    echo "Starting client 2"
    python3 runclient.py --hospital=2 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 3"
    python3 runclient.py --hospital=3 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 6"
    python3 runclient.py --hospital=6 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 8"
    python3 runclient.py --hospital=8 --seed=${j} --seer=${seer} &
    sleep 1
    
    echo "Starting client 9"
    python3 runclient.py --hospital=9 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 10"
    python3 runclient.py --hospital=10 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 11"
    python3 runclient.py --hospital=11 --seed=${j} --seer=${seer} &
    sleep 1

    echo "Starting client 12"
    python3 runclient.py --hospital=12 --seed=${j} --seer=${seer} &
    sleep 1

    wait
    
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
