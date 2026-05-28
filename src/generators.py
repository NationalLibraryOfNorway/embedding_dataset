import asyncio
import random
from datasets import Dataset
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

load_dotenv()

token = os.getenv("HF_TOKEN")

class Generator(ABC):

    def __init__(
            self, 
            model_id: str, 
            temperature: float, 
            top_p: float, 
            prompt: str, 
            language: str,
            samples: int,
            base_url: str = "http://localhost:8000/v1",
            api_key: str = "dummy",
            max_tokens: int = 8192,
            task: list=[], num_words: list=[], clarity: list=[], difficulty: list=[]
            ) -> None:
        
        self.model_id = model_id
        self.samples = samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        self.prompt = prompt

        self.language = language

        self.task = task
        self.num_words = num_words
        self.clarity = clarity
        self.difficulty = difficulty


    def set_prompts(self):
        self.prompts = [[{"role": "user", "content": self.make_prompt()}] for i in range(self.samples)]
        print(f"EXAMPLE PROMPT:\n\n{self.prompts[0]}")


    async def _generate_one(self, messages, semaphore, max_retries=3):
        extra_kwargs = {}
        if "gemini" in self.model_id.lower():
            extra_kwargs["extra_body"] = {"google": {"thinking_config": {"thinking_level": "low"}}}

        async with semaphore:
            for attempt in range(max_retries):
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    **extra_kwargs,
                )
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    return choice.message.content
                print(f"  Empty response (attempt {attempt + 1}/{max_retries}, finish_reason={choice.finish_reason})")
            print("  Skipping after max retries.")
            return None

    async def _generate_async(self):
        self.set_prompts()
        semaphore = asyncio.Semaphore(256)  # limit concurrency
        tasks = [self._generate_one(msgs, semaphore) for msgs in self.prompts]
        results = await asyncio.gather(*tasks)
        # Filter out failed generations
        paired = [(r, p) for r, p in zip(results, self.prompts) if r is not None]
        if paired:
            results, self.prompts = zip(*paired)
            results = list(results)
            self.prompts = list(self.prompts)
        else:
            results = []
            self.prompts = []
        dropped = self.samples - len(results)
        if dropped:
            print(f"  Dropped {dropped}/{self.samples} samples due to empty responses.")
        await self.client.close()
        return results

    def _generate(self):
        return asyncio.run(self._generate_async())

    def generate(self) -> Dataset:
        outputs = self._generate()
        outputs = self.post_process(outputs)
        return outputs
    
    @abstractmethod
    def make_prompt(self):
        pass

    def post_process(self, outputs: list[str]) -> Dataset:
        print(f"\n\nOUTPUT EXAMPLE:\n\n {outputs[0]}")

        df = pd.DataFrame({"response" : outputs})
        df["model"] = self.model_id
        df["prompt"] = self.prompts

        dataset = Dataset.from_pandas(df)
        return dataset


class GenerateFromTextClassificationTask(Generator):
    """
    Table 9
    """

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            num_words=random.choice(self.num_words),
            clarity=random.choice(self.clarity),
            difficulty=random.choice(self.difficulty),
            language=self.language
        )

        return _prompt
    
    
class GenerateFromRetrievalTask(Generator):

    """
    Table 8
    """

    def __init__(self, model_id: str, temperature: float, top_p: float, prompt: str, language: str, samples: int, task: list, num_words: list, clarity: list, difficulty: list, query_type: list, query_length: list, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy", max_tokens: int = 8192) -> None:
        super().__init__(model_id, temperature, top_p, prompt, language, samples, base_url=base_url, api_key=api_key, max_tokens=max_tokens, task=task, num_words=num_words, clarity=clarity, difficulty=difficulty)

        self.query_length = query_length
        self.query_type = query_type

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            query_type=random.choice(self.query_type),
            query_length=random.choice(self.query_length),
            clarity=random.choice(self.clarity),
            num_words=random.choice(self.num_words),
            difficulty=random.choice(self.difficulty),
            language=self.language
        )

        return _prompt    


class GenerateFromTextMatchingTask(Generator):

    """
    To be used for both table 10 and 11
    """

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            task=random.choice(self.task),
            language=self.language
        )

        return _prompt    

class GenerateUnitTriple(Generator):
    """
    Table 12: Prompt template for monolingual STS. For placeholders, "{high_score}" ∈ {4, 4.5, 5}, "{low_score}" ∈
    {2.5, 3, 3.5}, "{unit}" ∈ {sentence, phrase, passage}, "{difficulty}" ∈ {elementary school, high school, college}.
    """

    def __init__(self, model_id: str, temperature: float, top_p: float, prompt: str, language: str, samples: int, 
                 high_score, low_score, unit,
                 task: list = [], num_words: list = [], clarity: list = [], difficulty: list = [],
                 base_url: str = "http://localhost:8000/v1", api_key: str = "dummy", max_tokens: int = 8192) -> None:
        super().__init__(model_id, temperature, top_p, prompt, language, samples, base_url=base_url, api_key=api_key, max_tokens=max_tokens, task=task, num_words=num_words, clarity=clarity, difficulty=difficulty)

        self.unit = unit
        self.high_score = high_score
        self.low_score = low_score

    def make_prompt(self) -> dict:

        _prompt = self.prompt.format(
            unit=random.choice(self.unit),
            high_score=random.choice(self.high_score),
            low_score=random.choice(self.low_score),
            difficulty=random.choice(self.difficulty),
            language=self.language,
        )

        return _prompt
