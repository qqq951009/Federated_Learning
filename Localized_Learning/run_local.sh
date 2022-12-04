#!/bin/bash
# site_list = (2 3 6 8)
#declare -i j=43
seer=0
END=102
for ((j=42;j<=END;j++)); do
    echo "Start Local Training"
    
    # for d in 2 3 6 8; do
    #    echo $d
    #    python3 runlocal.py --hospital=${d} --seed=${j} --seer=${seer}
    #    wait
    # done

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
     
    wait

done
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
