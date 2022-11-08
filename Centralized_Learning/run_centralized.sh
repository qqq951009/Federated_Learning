#!/bin/bash
seer=0
END=102
for ((seed=43;seed<=END;seed++)); do
    python3 centralized.py --seed=${seed} --seer=${seer}
    wait
done
