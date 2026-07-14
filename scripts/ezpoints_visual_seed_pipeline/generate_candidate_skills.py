#!/usr/bin/env python3
"""Run the shared visual episode-skill generator for EZPoints rollouts."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sokoban_visual_seed_pipeline.generate_candidate_skills import main


if __name__ == "__main__":
    main()
