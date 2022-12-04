#!/bin/bash
# site_list = (2 3 6 8)
#declare -i j=43
END=43
for ((j=42;j<=END;j++)); do
    echo "Start Local Training"
    
    for d in 2 3 6 8; do
        echo $d
        python3 runlocal.py --hospital=${d} --seed=${j} --seer=${seer}
        # sleep 5
        wait
    done
        
    #python3 rundraw.py
done

