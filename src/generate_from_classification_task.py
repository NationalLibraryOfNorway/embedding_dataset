"""
Table 9: Prompt template for the long-short matching subgroup. For placeholders, “{num_words}” ∈ {"less than 10",
"at least 10", "at least 50", "at least 100", "at least 200"}, “{difficulty}” ∈ {high school, college, PhD}, “{clarity}” ∈
{clear, understandable with some effort, ambiguous}.
"""
from generators import GenerateFromTextClassificationTask
import variables
import argparse

from utils import load_task_dataset, format_language


def main(language: str):

    print(f"Will generate data in: {language}")


    task_dataset_id = "synthetic-from-classification-tasks"

    num_words = ["less than 10", "at least 10", "at least 50", "at least 100", "at least 200"]
    difficulty = ["high school", "college", "PhD"]
    clarity = ["clear", "understandable with some effort", "ambiguous"]

    task = load_task_dataset(f"NbAiLab/{variables.TEXT_CLASSIFICATION_TASK_DATASET_NAME}-processed")


    prompt = f"""You have been assigned a text classification task: {{task}}

    Your mission is to write one text classification example for this task in JSON format. The JSON object must contain the following keys:
    - "input_text": a string, the input text specified by the classification task.
    - "label": a string, the correct label of the input text.
    - "misleading_label": a string, an incorrect label that is related to the task.

    Please adhere to the following guidelines:
    - The "input_text" should be {{num_words}} words and diverse in expression.
    - The "misleading_label" must be a valid label for the given task, but not as appropriate as the "label" for the
    "input_text".
    - The values for all fields should be in {{language}}.
    - Avoid including the values of the "label" and "misleading_label" fields in the "input_text", that would make
    the task too easy.
    - The "input_text" is {{clarity}} and requires {{difficulty}} level education to comprehend.

    Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

    generator = GenerateFromTextClassificationTask(
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
        num_words=num_words,
        clarity=clarity,
        difficulty=difficulty
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