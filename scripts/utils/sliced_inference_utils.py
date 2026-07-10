import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.utils.sliced_inference_utils import *  # noqa: F401,F403
