#!/bin/bash

for i in {1..9}; do
	winpty python sim_assembly_mean.py $1 $i
done
