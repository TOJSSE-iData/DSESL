#!/bin/bash

nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 0 --cache_max_id 2499 >>./logs/cache_20240423_0-2499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 2500 --cache_max_id 4999 >>./logs/cache_20240423_2500-4999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 5000 --cache_max_id 7499 >>./logs/cache_20240423_5000-7499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 7500 --cache_max_id 9999 >>./logs/cache_20240423_7500-9999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 10000 --cache_max_id 12499 >>./logs/cache_20240423_10000-12499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 12500 --cache_max_id 14999 >>./logs/cache_20240423_12500-14999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 15000 --cache_max_id 17499 >>./logs/cache_20240423_15000-17499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 17500 --cache_max_id 19999 >>./logs/cache_20240423_17500-19999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 20000 --cache_max_id 22499 >>./logs/cache_20240423_20000-22499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 22500 --cache_max_id 24999 >>./logs/cache_20240423_22500-24999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 25000 --cache_max_id 27499 >>./logs/cache_20240423_25000-27499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 27500 --cache_max_id 29999 >>./logs/cache_20240423_27500-29999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 30000 --cache_max_id 32499 >>./logs/cache_20240423_30000-32499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 32500 --cache_max_id 34999 >>./logs/cache_20240423_32500-34999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 35000 --cache_max_id 37499 >>./logs/cache_20240423_35000-37499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 37500 --cache_max_id 39999 >>./logs/cache_20240423_37500-39999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 40000 --cache_max_id 42499 >>./logs/cache_20240423_40000-42499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 42500 --cache_max_id 44999 >>./logs/cache_20240423_42500-44999.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 45000 --cache_max_id 47499 >>./logs/cache_20240423_45000-47499.log 2>&1 &
nohup python -u struct_gnn/cache_script.py --device cpu --cache_min_id 47500 --cache_max_id 49999 >>./logs/cache_20240423_47500-49999.log 2>&1 &
