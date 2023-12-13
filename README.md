# Instructions to Run Our Code
## Part 1
With quantization: run `python run_part1_with_quant.py`

Without quantization: run `python run_part1_no_quant.py`

Standard vs Speculative decoding: run `python run_part1_decoding.py`

## Part 2
Protocol A: 

run `python run_part2_a.py --prune_method='individual'` for individual weights pruning and `python run_part2_a.py --prune_method='l2norm'` for structured pruning

Protocol B:
run `python run_part2_b.py --prune_method='individual'` for individual weights pruning and `python run_part2_b.py --prune_method='l2norm'` for structured pruning

## Leaderboard
run `python run_experiments.py` which generates `results.json`