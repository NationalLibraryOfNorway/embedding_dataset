"""
Table 12: Prompt template for monolingual STS. For placeholders, “{high_score}” ∈ {4, 4.5, 5}, “{low_score}” ∈
{2.5, 3, 3.5}, “{unit}” ∈ {sentence, phrase, passage}, “{difficulty}” ∈ {elementary school, high school, college}.
"""

from generators import GenerateUnitTriple
from utils import format_language
import variables
import argparse

def main(language: str):

    print(f"Will generate data in: {language}")

    task = ["task1", "task2"]
    unit = ["sentence", "phrase", "passage"]
    high_score = ["4", "4.5", "5"]
    low_score = ["2.5", "3", "3.5"]
    difficulty = ["elementary school", "high school", "college"]

    prompt = f"""Write a {{unit}} triple with varying semantic similarity scores in JSON format. The semantic similarity score ranges from 1 to
    5, with 1 denotes least similar and 5 denotes most similar.
    Please adhere to the following guidelines:
    - The keys in JSON are "S1", "S2", and "S3", the values are all strings in {{language}}, do not add any other keys.
    - There should be some word overlaps between all three {{unit}}s.
    - The similarity score between S1 and S2 should be {{high_score}}.
    - The similarity score between S1 and S3 should be {{low_score}}.
    - The {{unit}}s require {{difficulty}} level education to understand and should be diverse in terms of topic and length.
    Your output must always be a JSON object only with three keys "S1", "S2" and "S3", do not explain yourself or output
    anything else. Be creative!"""

    generator = GenerateUnitTriple(
        model_id=variables.MODEL_ID,
        temperature=variables.TEMPERATURE,
        top_p=variables.TOP_P,
        prompt=prompt, 
        language=language,
        task=task,
        samples=variables.TOTAL_DESIRED_SAMPLES,
        base_url=variables.BASE_URL,
        api_key=variables.API_KEY,
        max_tokens=variables.MAX_TOKENS,
        unit=unit,
        high_score=high_score,
        difficulty=difficulty,
        low_score=low_score
        )

    dataset = generator.generate()

    lang_slug = format_language(language)

    try:
        dataset.to_csv(variables.OUTPUT_DIR / f"{variables.TASK_DATASET_ID_UNIT_TRIPLE}-{lang_slug}.csv", index=False)

    except Exception as e:

        print(f"could not save {variables.TASK_DATASET_ID_UNIT_TRIPLE}-{lang_slug}.csv")

    if variables.PUSH_TO_HF:
        dataset.push_to_hub(f"NbAiLab/{variables.TASK_DATASET_ID_UNIT_TRIPLE}-{lang_slug}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a language argument.")

    parser.add_argument("language", type=str, help="The language of the generated data.")

    args = parser.parse_args()

    main(language=args.language)