#!/bin/bash

n_nodes=$1
offset=$2
echo "Launching $n_nodes nodes"
rank=0
mkdir -p logs/launch_stdout
mkdir -p logs/launch_stderr
master=$(cat ~/computers_name | tail -n +$offset | head -n 1 | cut -d @ -f 2)
port=$(shuf -i 2000-65000 -n 1)
for i in $(cat ~/computers_name | tail -n +$offset | head -n $n_nodes)
do
    rm -f logs/launch_stderr/log.$i
    rm -f logs/launch_stdout/log.$i
    ssh -oStrictHostKeyChecking=no $i "tmux new-session -d -s imagnum \"cd ~/timothee/imagnum/let_there_be_color && ./launch_worker.sh $port $master $n_nodes $rank >logs/launch_stdout/log.$i 2>logs/launch_stderr/log.$i\"" &
    echo Launched node $i with rank $rank
    ((rank++))
done
