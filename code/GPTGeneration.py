import os
import tiktoken
from openai import OpenAI

class GPTGeneration():
    def __init__(
            self, model_name="gpt-4o-mini", max_tokens=16, temperature=0.7
            ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = 1130

        # invariant variables
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        # model keys
        self.key = os.getenv("OPENAI_KEY")
        self.client = OpenAI(api_key=self.key)
        self.seed = 1130

    # Returns the response from the model given a system message and a prompt text.
    def get_response(self, prompt_text, judge=True):   
        # Example message: "You are a helpful assistant who can extract verifiable atomic claims from a piece of text."
        if judge:
            message = [
                {"role": "system", "content": "You are a helpful assistant who is good at judging whether Answer 1 is entailed in Answer 2."},
                {"role": "user", "content": prompt_text}
            ]
        else:
            message = [
                {"role": "user", "content": prompt_text}
            ]
        response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    seed=self.seed,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    )
        response_content = response.choices[0].message.content.strip()

        # count tokens in prompt and response
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, prompt_tokens, response_tokens
    
    # Returns the number of tokens in a text string.
    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens
