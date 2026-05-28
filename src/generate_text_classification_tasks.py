"""
Implements table 9 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

def main():
    prompt = f"""Brainstorm a list of potentially useful text classification tasks.
    Please adhere to the following guidelines:
    - Tasks should cover a diverse range of domains and task types.
    Your output must always be a Python list of strings only, with about 20 elements, and each element corresponds to a distinct text classification task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    csv_save_as = f"{variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME}.csv"

    generate_task(total_desired_samples=variables.TOTAL_DESIRED_TASKS,
                  model_id=variables.MODEL_ID,
                  prompt=prompt,
                  csv_save_as=csv_save_as,
                  base_url=variables.BASE_URL,
                  api_key=variables.API_KEY,
                  push_to_hf=variables.PUSH_TO_HF,
                  hf_dataset_name=f"{variables.HF_USER}/{variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME}",
                  output_dir=variables.OUTPUT_DIR)

if __name__ == "__main__":
    main()