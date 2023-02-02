#!/bin/bash
# site_list = (2 3 6 8)
#declare -i j=43

echo "Start Local Training"
# for s in 42 53 79 81; do 
#     for d in 3 ; do
#         echo $d
#         python3 runlocal.py --hospital=${d} --seed=${s}
#         wait
#     done
# done


END=102
for ((j=43;j<=END;j++)); do
    echo "Start Local Training"
    
    for d in 2 3 6 8 9 10 11 12 ; do # 9 10 11 12
        echo $d
        python3 test_local.py --hospital=${d} --seed=${j}
        # sleep 5
        wait
    done
done

