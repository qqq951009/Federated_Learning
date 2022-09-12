#!/bin/bash
declare -i seer=0
average_weight='average'
END=102
for ((seed=42;seed<=END;seed++)); do
    echo "Start Training Pretrained Model"
    python3 transfer_learning_pretrained.py --seed=${seed} --seer=$seer --encode_dict=${average_weight}
    sleep 1
    
    echo "site 2 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=2 --encode_dict=${average_weight}
    sleep 1

    echo "site 6 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=6 --encode_dict=${average_weight}
    sleep 1

    echo "site 8 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=8 --encode_dict=${average_weight}
    sleep 1

    if (($seer == 1));
    then
        echo "site 3 Finetune Model"
        python3 transfer_learning_finetune.py --seed=${seed} --seer=$seer --hospital=3 --encode_dict=${average_weight}
        sleep 1
    fi


done