#!/bin/bash
declare -i seer=0
END=44
for ((seed=43;seed<=END;seed++)); do
    python3 make_encode_map.py --seed=${seed} --seer=$seer
    echo "Start Training Pretrained Model"
    python3 transfer_learning_pretrained.py --seed=${seed} --seer=$seer
    
    echo "site 2 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=2
    sleep 1

    echo "site 6 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=6
    sleep 1

    echo "site 8 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=8 
    sleep 1

    if (($seer == 1));
    then
        echo "site 3 Finetune Model"
        python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=3
        sleep 1
    fi


done