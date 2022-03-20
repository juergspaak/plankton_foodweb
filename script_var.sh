#!/bin/bash

for i in {1..9}; do
	winpty python sim_assembly_variance.py
done

winpty python sim_assembly_corr.py

winpty python sim_assembly_long.py