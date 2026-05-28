from datasets import Dataset, load_dataset
from openai import OpenAI
from pathlib import Path


def format_language(language: str) -> str:
    """Format language string for use in filenames and dataset IDs."""
    return language.lower().replace("(", "").replace(")", "").replace(" ", "_")


def load_task_dataset(dataset_name: str) -> list[str]:
    """Load task definitions from local CSV (output dir) or HF Hub."""
    import variables
    local_name = dataset_name.split("/")[-1]
    local_path = variables.OUTPUT_DIR / f"{local_name}.csv"
    if local_path.exists():
        ds = Dataset.from_csv(str(local_path))
        return list(ds["response"])
    try:
        ds = load_dataset(dataset_name)
        return list(ds["train"]["response"])
    except Exception:
        raise FileNotFoundError(
            f"Could not load task dataset '{dataset_name}' from the Hub or locally as '{local_path}'. "
            f"Run Phase 1 (generate_tasks.py + post_processing.py) first."
        )


def generate_task(
        total_desired_samples: int,
        model_id: str,
        prompt: str,
        csv_save_as: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        push_to_hf: bool=False,
        hf_dataset_name: str="",
        output_dir: Path = Path("output")
        ):

    samples = max(1, total_desired_samples // 20) # each prompt will result in approximately 20 topics

    client = OpenAI(base_url=base_url, api_key=api_key)

    prompts_list = [prompt for _ in range(samples)]

    responses = []
    extra_kwargs = {}
    if "gemini" in model_id.lower():
        extra_kwargs["extra_body"] = {"google": {"thinking_config": {"thinking_level": "low"}}}

    for p in prompts_list:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": p}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=8192,
            **extra_kwargs,
        )
        responses.append(response.choices[0].message.content)

    print(responses)

    responses = [{"response" : response} for response in responses]

    dataset = Dataset.from_list(responses)

    dataset.to_csv(output_dir / csv_save_as, index=False)

    if push_to_hf:

        dataset.push_to_hub(hf_dataset_name)


