from datasets import load_dataset
import variables


def inspect_tasks():
    dataset_ids = [
        f"NbAiLab/{variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME}-processed",
        f"NbAiLab/{variables.RETRIEVAL_TASK_DATASET_NAME}-processed",
        f"NbAiLab/{variables.TEXT_MATCHING_LONG_DATASET_NAME}-processed",
        f"NbAiLab/{variables.TEXT_MATCHING_SHORT_DATASET_NAME}-processed",
    ]

    for data_id in dataset_ids:
        data = load_dataset(data_id)
        print(data)
        print(data["train"]["response"][0])


def inspect_data():
    dataset_ids = [
        "NbAiLab/synthetic-from-retrieval-tasks",
        "NbAiLab/synthetic-from-unit-triple-tasks",
        "NbAiLab/synthetic-from-classification-tasks",
        "NbAiLab/synthetic-from-text-matching-long-tasks",
        "NbAiLab/synthetic-from-text-matching-short-tasks",
    ]

    for data_id in dataset_ids:
        data = load_dataset(data_id)
        print(data)
        print(data["train"]["prompt"][0])
        print(data["train"]["response"][0])
        print(data["train"]["model"][0])


def main():
    inspect_tasks()
    inspect_data()


if __name__ == "__main__":
    main()