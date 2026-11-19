#!/bin/bash
# Zero-shot video baseline (Qwen3-VL-4B, both egocentric views, 32 s at 1 fps) on one V100 or better.
# Usage: sbatch -J cs7170-video -p gpu --gres=gpu:1 --exclude=d1025,<your busy nodes> -c 4 --mem=48G -t 06:00:00 \
#          -o /scratch/basu.ang/cs7170/logs/video_%j.out video_zeroshot.sh [OUT] [extra args, e.g. --limit 2]
# Set ITEMS=path (sbatch --export=ALL,ITEMS=...) to score a subset, e.g. a smoke test.
set -euo pipefail
S=/scratch/basu.ang/cs7170
export HF_HOME=$S/hf XDG_CACHE_HOME=$S/cache HF_HUB_OFFLINE=1
OUT=${1:-$S/runs/mc_v2/video_w32}
shift || true
cd $S
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 300 &
envs/cs7170/bin/python -I code/blind_llm.py --items ${ITEMS:-runs/mc_v2/items.jsonl} --graphs labels/recipes/graphs \
  --frames runs/mc_v2/frames_w32 --px 320 --batch 2 --out "$OUT" "$@"
kill %1 || true
