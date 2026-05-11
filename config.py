import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "Train.csv"
VAL_CSV = DATA_DIR / "Val.csv"
TEST_CSV = DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "SampleSubmission.csv"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Claude model to use for generation (haiku = fast + cheap, sonnet = better quality)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Language map: subset code → human readable
LANGUAGE_MAP = {
    "Aka_Gha": "Akan (Ghanaian Twi)",
    "Lug_Uga": "Luganda (Ugandan)",
    "Swa_Ken": "Kiswahili (Kenyan)",
    "Amh_Eth": "Amharic (Ethiopian)",
    "Eng_Uga": "English (Uganda)",
    "Eng_Gha": "English (Ghana)",
    "Eng_Eth": "English (Ethiopia)",
    "Eng_Ken": "English (Kenya)",
}

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
