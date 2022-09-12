#!/bin/bash

#declare -i j=42
seer=1
average_weight='average'
END=102
for ((j=44;j<=END;j++)); do


    echo "Starting seed" ${j}
    python3 runserver.py --seed=${j} --seer=${seer} &
    sleep 1  # Sleep for 3s to give the server enough time to start


    echo "Starting client 2"
    python3 runclient_new.py --hospital=2 --seed=${j} --seer=${seer} --encode_dict=${average_weight} &
    sleep 1

    echo "Starting client 3"
    python3 runclient_new.py --hospital=3 --seed=${j} --seer=${seer} --encode_dict=${average_weight} &
    sleep 1

    echo "Starting client 6"
    python3 runclient_new.py --hospital=6 --seed=${j} --seer=${seer} --encode_dict=${average_weight} &
    sleep 1

    echo "Starting client 8"
    python3 runclient_new.py --hospital=8 --seed=${j} --seer=${seer} --encode_dict=${average_weight} &
    sleep 1
    
    if (($seer == 1));
    then
        echo "Starting client 9"
        python3 runclient_new.py --hospital=9 --seed=${j} --seer=${seer} --encode_dict=${average_weight} &
        sleep 1
    fi
    wait
    
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
