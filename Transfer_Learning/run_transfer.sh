#!/bin/bash
declare -i seer=0
END=102
for ((seed=44;seed<=END;seed++)); do
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

    echo "site 9 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=9
    sleep 1

    echo "site 10 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=10 
    sleep 1

    echo "site 11 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=11
    sleep 1

    echo "site 12 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=12 
    sleep 1

    if (($seer == 1));
    then
        echo "site 3 Finetune Model"
        python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=3
        sleep 1
    fi


done