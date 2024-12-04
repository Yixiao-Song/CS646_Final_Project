import os
import json
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

    def get_response(self, prompt_text, judge=True, return_json=False):   
        """
        Returns the response from the model given a prompt text.
        Args:
            prompt_text (str): The input text to process
            judge (bool): Whether to use the judging system prompt
            response_format (str): Either "text" or "json" to specify output format
        Returns:
            tuple: (response_content, prompt_tokens, response_tokens)
        """
        if judge:
            message = [
                {"role": "system", "content": "You are a helpful assistant who is good at judging whether Answer 1 is entailed in Answer 2. Return your response as a JSON object with fields for 'is_entailed' (boolean) and 'explanation' (string)."},
                {"role": "user", "content": prompt_text}
            ]
        else:
            message = [
                {"role": "user", "content": prompt_text}
            ]

        # Configure response format
        api_params = {
            "model": self.model_name,
            "messages": message,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if return_json:
            api_params["response_format"] = {"type": "json_object"}
            
        try:
            response = self.client.chat.completions.create(**api_params)
            response_content = response.choices[0].message.content.strip()
            
            # If JSON format was requested, verify the response is valid JSON
            if return_json:
                try:
                    response_content = json.loads(response_content)
                except json.JSONDecodeError:
                    raise ValueError("Model returned invalid JSON response")

            # count tokens in prompt and response
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            response_tokens = len(self.tokenizer.encode(str(response_content)))
            return response_content, prompt_tokens, response_tokens

        except Exception as e:
            raise Exception(f"Error getting response from OpenAI API: {str(e)}")
    
    def tok_count(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens
