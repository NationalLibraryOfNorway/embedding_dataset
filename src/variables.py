import os
import tomllib
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_config_path = Path(os.getenv("CONFIG_PATH", _project_root / "configs" / "local.toml"))
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

# Model
MODEL_ID = _config["model"]["model_id"]
TEMPERATURE = _config["model"]["temperature"]
TOP_P = _config["model"]["top_p"]
MAX_TOKENS = _config["model"].get("max_tokens", 8192)

# Server
_auth_mode = _config["server"].get("auth", "api_key")

if _auth_mode == "gcloud":
    import google.auth
    import google.auth.transport.requests

    _credentials, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    _credentials.refresh(google.auth.transport.requests.Request())
    API_KEY = _credentials.token

    _project_id = _config["server"].get("project_id") or _project or os.getenv("GOOGLE_CLOUD_PROJECT", "")
    _location = _config["server"].get("location", "global")
    BASE_URL = _config["server"]["base_url"].format(project_id=_project_id, location=_location)
else:
    BASE_URL = os.getenv("VLLM_BASE_URL", _config["server"]["base_url"])
    API_KEY = os.getenv("VLLM_API_KEY", _config["server"]["api_key"])

# Hugging Face
HF_USER = _config["huggingface"]["hf_user"]
PUSH_TO_HF = _config["huggingface"]["push_to_hf"]

# Generation
TOTAL_DESIRED_SAMPLES = _config["generation"]["total_desired_samples"]
TOTAL_DESIRED_TASKS = TOTAL_DESIRED_SAMPLES // 2
OUTPUT_DIR = Path(_config["generation"].get("output_dir", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset names
TEXT_CLASSIFICATION_TASK_DATASET_NAME = _config["datasets"]["text_classification_task_dataset_name"]
RETRIEVAL_TASK_DATASET_NAME = _config["datasets"]["retrieval_task_dataset_name"]
TEXT_MATCHING_SHORT_DATASET_NAME = _config["datasets"]["text_matching_short_dataset_name"]
TEXT_MATCHING_LONG_DATASET_NAME = _config["datasets"]["text_matching_long_dataset_name"]
TASK_DATASET_ID_UNIT_TRIPLE = _config["datasets"]["task_dataset_id_unit_triple"]
