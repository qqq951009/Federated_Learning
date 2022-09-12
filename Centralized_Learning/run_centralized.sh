#!/bin/bash
seer=1
END=43
for ((seed=42;seed<=END;seed++)); do
    python3 centralized.py --seed=${seed} --seer=${seer}
    #sleep 1
done