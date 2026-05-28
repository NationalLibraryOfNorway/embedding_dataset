"""
Table 10: Prompt template for the short-short matching subgroup. We do not generate negative documents as the
matching task is already reasonably difficult.
"""

from generators import GenerateFromTextMatchingTask
from utils import load_task_dataset, format_language
import variables
import argparse



def main(language: str):

    print(f"Will generate data in: {language}")

    task_dataset_id = "synthetic-from-text-matching-short-tasks"

    task = load_task_dataset(f"NbAiLab/{variables.TEXT_MATCHING_SHORT_DATASET_NAME}-processed")


    prompt = f"""You have been assigned a text matching task: {{task}}
    Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
    - "input": a string, a random input specified by the task.
    - "positive_document": a string, a relevant document for the "input" according to the task.
    Please adhere to the following guidelines:
    - The values of all fields should be in {{language}}.
    - Both the "input" and "positive_document" should be very short (a sentence or a phrase), avoid substantial word overlaps,
    otherwise the task would be too easy.
    - The "input" and "positive_document" should be independent of each other.
    Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

    generator = GenerateFromTextMatchingTask(
        model_id=variables.MODEL_ID,
        temperature=variables.TEMPERATURE,
        top_p=variables.TOP_P,
        prompt=prompt, 
        language=language,
        samples=variables.TOTAL_DESIRED_SAMPLES,
        base_url=variables.BASE_URL,
        api_key=variables.API_KEY,
        max_tokens=variables.MAX_TOKENS,
        task=task,
        )

    dataset = generator.generate()

    lang_slug = format_language(language)

    try:
        dataset.to_csv(variables.OUTPUT_DIR / f"{task_dataset_id}-{lang_slug}.csv", index=False)

    except Exception as e:

        print(f"could not save {task_dataset_id}-{lang_slug}.csv")
        print(f"Exception: {e}")

    if variables.PUSH_TO_HF:
        dataset.push_to_hub(f"NbAiLab/{task_dataset_id}-{lang_slug}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a language argument.")

    parser.add_argument("language", type=str, help="The language of the generated data.")

    args = parser.parse_args()

    main(language=args.language)