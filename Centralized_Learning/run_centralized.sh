#!/bin/bash

END=98
for ((seed=42;seed<=END;seed++)); do
    python3 centralized.py --seed=${seed}
    sleep 3
done