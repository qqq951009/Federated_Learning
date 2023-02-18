#!/bin/bash
declare -i seer=0
END=102
for ((seed=42;seed<=END;seed++)); do
    # python3 make_encode_map.py --seed=${seed} --seer=$seer
    echo "Start Training Pretrained Model"
    python3 testtl_pretraianed.py --seed=${seed}
    
    echo "site 2 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=2
    sleep 1

    echo "site 6 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=6
    sleep 1

    echo "site 8 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=8 
    sleep 1

    echo "site 9 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=9
    sleep 1

    echo "site 10 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=10 
    sleep 1

    echo "site 11 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=11
    sleep 1

    echo "site 12 Finetune Model"
    python3 testtl_finetune.py --seed=${seed} --hospital=12 
    sleep 1


done