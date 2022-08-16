#!/bin/bash

END=98
for ((seed=45;seed<=END;seed++)); do
    echo "Start Training Pretrained Model"
    python3 transfer_learning_pretrained.py --seed=${seed}
    sleep 5

    echo "site 2 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --hospital=2
    sleep 3

    echo "site 6 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --hospital=6
    sleep 3

    echo "site 8 Finetune Model"
    python3 transfer_learning_finetune.py --seed=${seed} --hospital=8
    sleep 3
done