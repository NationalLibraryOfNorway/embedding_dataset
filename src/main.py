import time

import generate_all_languages
import generate_tasks
import variables


def main():
    # Skip task generation if processed task files already exist
    task_names = [
        variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME,
        variables.RETRIEVAL_TASK_DATASET_NAME,
        variables.TEXT_MATCHING_LONG_DATASET_NAME,
        variables.TEXT_MATCHING_SHORT_DATASET_NAME,
    ]
    all_exist = all(
        (variables.OUTPUT_DIR / f"{name}-processed.csv").exists()
        for name in task_names
    )

    if all_exist:
        print("Task files already exist, skipping Phase 1.")
    else:
        t0 = time.perf_counter()
        generate_tasks.main()
        t1 = time.perf_counter()
        print(f"Generated tasks in {t1 - t0} seconds.")

    t0 = time.perf_counter()
    generate_all_languages.main()
    t1 = time.perf_counter()
    print(f"Generated all languages in {t1 - t0} seconds.")


if __name__ == "__main__":
    main()
