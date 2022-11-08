#!/bin/bash
# site_list = (2 3 6 8)
#declare -i j=43
seer=0
END=102
for ((j=42;j<=END;j++)); do
    echo "Start Local Training"
    #for site in ${site_list[@]}; do
    #    python3 runlocal.py --hospital=site --seed=${j} --seer=0
    #    sleep 1

    echo "Start client 2" 
    python3 runlocal.py --hospital=2 --seed=${j} --seer=${seer}

    sleep 5
    echo "Start client 3"

    python3 runlocal.py --hospital=3 --seed=${j} --seer=${seer}

    sleep 5
    echo "Start client 6"
    python3 runlocal.py --hospital=6 --seed=${j} --seer=${seer}

    sleep 5
    echo "Start client 8"
    python3 runlocal.py --hospital=8 --seed=${j} --seer=${seer}
    

    #sleep 5
    #echo "Start client 9" 
    #python3 runlocal.py --hospital=9 --seed=${j}
    wait
    #python3 rundraw.py
done
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
