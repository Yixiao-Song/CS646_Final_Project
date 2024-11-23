import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class GetResponse:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        """Initializes the Qwen2.5-14B-Instruct model and tokenizer.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.model_name = model_name

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Model loaded.")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            )
        self.tokenizer.padding_side = 'left'
        print("Tokenizer loaded.")

    def get_response(self, prompt, max_new_tokens=512):
        """Generates response from Qwen2.5-14B-Instruct.

        Args:
            prompt (str): The user prompt to generate a response for.
            max_new_tokens (int): The maximum number of new tokens to generate. 

        Returns:
            str: The generated response.
        """
        # Prepare the chat messages
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Apply the chat template and tokenize the messages
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Convert text into model inputs
        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
            ).to(self.model.device)

        # Generate tokens
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

        # Exclude input tokens from generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids \
                in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode and return the response
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
            )[0]
        return response

    def get_batch_responses(self, prompts, max_new_tokens=512):
        """Generates responses for a batch of prompts.

        Args:
            prompts (list): A list of user prompts.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            responses (list): A list of generated responses.
        """
        # Prepare the batched inputs
        messages_list = [
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
             {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        # Apply the chat template for each message
        texts = [self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in messages_list]

        # Tokenize all texts
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Generate batched tokens
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

        # Decode all generated responses
        responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return responses
