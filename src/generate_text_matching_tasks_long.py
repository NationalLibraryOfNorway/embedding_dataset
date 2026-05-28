"""
Implements table 10 in https://arxiv.org/pdf/2401.00368
"""

from utils import generate_task
import variables

def main():
    prompt = f"""Brainstorm a list of text matching tasks where the queries are long documents.
    Here are a few examples:
    - Given a document that supports a debatable argument, find another document that contains opposite arguments.
    - Provided a lengthy business proposal, retrieve competitive business strategies in the same industry.
    Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct
    task in one sentence. Do not explain yourself or output anything else. Be creative!"""

    csv_save_as = f"{variables.TEXT_MATCHING_LONG_DATASET_NAME}.csv"

    generate_task(total_desired_samples=variables.TOTAL_DESIRED_TASKS,
                  model_id=variables.MODEL_ID,
                  prompt=prompt,
                  csv_save_as=csv_save_as,
                  base_url=variables.BASE_URL,
                  api_key=variables.API_KEY,
                  push_to_hf=variables.PUSH_TO_HF,
                  hf_dataset_name=f"{variables.HF_USER}/{variables.TEXT_MATCHING_LONG_DATASET_NAME}",
                  output_dir=variables.OUTPUT_DIR)

if __name__ == "__main__":
    main()