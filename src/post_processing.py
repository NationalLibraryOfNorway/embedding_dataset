"""
Post processing of the generated tasks
"""

from datasets import load_dataset, Dataset
from pathlib import Path
import variables


def load_task_data(dataset_name: str) -> list[str]:
    """Load raw task data from local CSV (output dir) or HF Hub."""
    local_path = variables.OUTPUT_DIR / f"{dataset_name}.csv"
    if local_path.exists():
        ds = Dataset.from_csv(str(local_path))
        return list(ds["response"])
    else:
        ds = load_dataset(f"{variables.HF_USER}/{dataset_name}")
        return list(ds["train"]["response"])


def save_processed(data: list[str], dataset_name: str):
    """Save processed task data locally and optionally push to Hub."""
    ds = Dataset.from_dict({"response": data})
    ds.to_csv(str(variables.OUTPUT_DIR / f"{dataset_name}-processed.csv"), index=False)
    if variables.PUSH_TO_HF:
        ds.push_to_hub(f"{variables.HF_USER}/{dataset_name}-processed")


def split_and_clean(entry):
    """Splits a single dataset entry into multiple cleaned entries."""
    sentences = entry.split("\n")
    cleaned_sentences = [sentence.lstrip("-").strip() for sentence in sentences if sentence.strip()]
    return cleaned_sentences


def process_dataset(responses: list[str]) -> list[str]:
    """Processes responses by splitting and cleaning entries."""
    all_sentences = []
    for entry in responses:
        all_sentences.extend(split_and_clean(entry))
    return all_sentences


def post_process_classification_tasks():
    data_list = load_task_data(variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME)

    new_data = []
    for i in data_list:
        i = i.replace("```", "").replace("```python", "").replace("[", "").replace("]", "")
        i = i.strip("'").lstrip('- ').strip()
        i = i.replace("'", "")
        new_data.extend(i.split("\n"))

    new_data = [n for n in new_data if n]
    save_processed(new_data, variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME)


def main():

    post_process_classification_tasks()

    dataset_names = [
        variables.RETRIEVAL_TASK_DATASET_NAME,
        variables.TEXT_MATCHING_LONG_DATASET_NAME,
        variables.TEXT_MATCHING_SHORT_DATASET_NAME,
    ]

    for name in dataset_names:
        responses = load_task_data(name)
        processed = process_dataset(responses)
        save_processed(processed, name)


if __name__ == "__main__":
    main()
