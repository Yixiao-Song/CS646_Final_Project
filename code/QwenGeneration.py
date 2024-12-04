import os
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class QwenGeneration:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        """Initializes the Qwen model and tokenizer with vLLM backend.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.model_name = model_name
        
        self.model = LLM(model=model_name)
        print("Model loaded.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'
        print("Tokenizer loaded.")
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512
        )

    def get_response(self, prompt, max_new_tokens=512):
        """Generates response(s) using vLLM backend.
        Args:
            prompt (Union[str, List[str]]): Single prompt or list of prompts to generate responses for.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            Union[str, List[str]]: Single response string if input was a string, 
                                  list of response strings if input was a list.
        """
        self.sampling_params.max_tokens = max_new_tokens
        
        # Convert single prompt to list for uniform processing
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else prompt
        
        # Process all prompts and check token counts
        texts = []
        token_counts = []
        valid_indices = []  # Track indices of valid prompts
        
        for i, p in enumerate(prompts):
            messages = [
                {"role": "user", "content": p}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Check token count
            num_tokens = len(self.tokenizer.encode(text))
            token_counts.append(num_tokens)
            
            if num_tokens <= (131072 - max_new_tokens):  # Keep track of valid prompts
                valid_indices.append(i)
                texts.append(text)
            else:
                print(f"Skipping prompt {i} due to token count: {num_tokens}")
        
        # Generate responses for valid prompts
        outputs = self.model.generate(texts, self.sampling_params) if texts else []
        
        # Reconstruct responses array maintaining original ordering
        responses = ["SKIPPED"] * len(prompts)  # Initialize all as skipped
        for valid_idx, output in zip(valid_indices, outputs):
            responses[valid_idx] = output.outputs[0].text
        
        # Return single string if input was single string
        return responses[0] if single_input else responses


# class QwenGeneration:
#     def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
#         """Initializes the Qwen2.5-14B-Instruct model and tokenizer.

#         Args:
#             model_name (str): The name of the model to load from Hugging Face.
#         """
#         self.model_name = model_name

#         # Load the model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype="auto",
#             device_map="auto",
#         )
#         print("Model loaded.")

#         # Load the tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             )
#         self.tokenizer.padding_side = 'left'
#         print("Tokenizer loaded.")

#     def get_response(self, prompt, max_new_tokens=512):
#         """Generates response from Qwen2.5-14B-Instruct.

#         Args:
#             prompt (str): The user prompt to generate a response for.
#             max_new_tokens (int): The maximum number of new tokens to generate. 

#         Returns:
#             str: The generated response.
#         """
#         # Prepare the chat messages
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]

#         # Apply the chat template and tokenize the messages
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         # Convert text into model inputs
#         model_inputs = self.tokenizer(
#             [text], return_tensors="pt"
#             ).to(self.model.device)

#         # Generate tokens
#         generated_ids = self.model.generate(
#             **model_inputs,
#             max_new_tokens=max_new_tokens,
#         )

#         # Exclude input tokens from generated tokens
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids \
#                 in zip(model_inputs.input_ids, generated_ids)
#         ]

#         # Decode and return the response
#         response = self.tokenizer.batch_decode(
#             generated_ids, skip_special_tokens=True
#             )[0]
#         return response
