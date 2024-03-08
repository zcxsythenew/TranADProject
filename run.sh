#!/bin/bash
for i in $(seq 1 5)
do
echo Run ${i}
python main.py --model TranAD --dataset k8s --retrain
mv words.pkl words_${i}.pkl
done