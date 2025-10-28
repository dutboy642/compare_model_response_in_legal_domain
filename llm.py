from openai import OpenAI
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
# local_api_key= os.getenv("LOCAL_API_KEY", "EMPTY")
# local_base_url= "http://localhost:8881/v1"
class llm:
    def __init__(self, model_name: str):
        self.client = self.get_llm(model_name)
        if model_name.lower()=="local":
            self.model_name= os.getenv("LOCAL_MODEL_NAME")
        else:
            self.model_name = model_name

    def get_llm(self, model_name):
        api_key_name= f"{model_name}_API_KEY"
        base_url_name= f"{model_name}_BASE_URL"

        api_key= os.getenv(api_key_name)
        base_url= os.getenv(base_url_name, None)
        print("Using API Key from:", api_key_name)
        print("Using Base URL from:", base_url_name)
        if "gemini" in model_name.lower():
            return self.create_gemini_client(api_key, base_url)
        return self.create_openai_client(api_key, base_url)

    def create_gemini_client(self, api_key: str, base_url: str):
        return genai.Client(api_key=api_key)

    def create_openai_client(self,api_key:str, base_url:str) -> OpenAI:
        """Create an OpenAI client configured to connect to a local server."""
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(self, prompt: str, max_tokens: int = 500):
        """Generate text using the local LLM."""
        if "gpt" in self.model_name.lower():
            response = self.client.responses.create(
            model=self.model_name,
            reasoning={ "effort": "low" },
            input=prompt,
            max_output_tokens=max_tokens,
        )
        elif "gemini" in self.model_name.lower():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
        else:    
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
            )

        return response
    
    def calculate_price(self, response)-> float:
        """Calculate the cost of the API call based on token usage."""
        # Example pricing: $0.0004 per 1K prompt tokens, $0.0006 per 1K completion tokens
        prompt_cost_per_1m_name= f"{self.model_name}_PROMPT_COST_PER_1M"
        completion_cost_per_1m_name= f"{self.model_name}_COMPLETION_COST_PER_1M"
        prompt_cost_per_1m= float(os.getenv(prompt_cost_per_1m_name,0.00))
        completion_cost_per_1m= float(os.getenv(completion_cost_per_1m_name,0.00))
        prompt_cost_per_1k = prompt_cost_per_1m/1000
        completion_cost_per_1k = completion_cost_per_1m/1000
        # get token usage from response
        if "gpt" in self.model_name.lower():
            prompt_tokens=response.usage.input_tokens
            completion_tokens=response.usage.output_tokens
        elif "gemini" in self.model_name.lower():
            prompt_tokens=response.usage_metadata.candidates_token_count
            completion_tokens=response.usage_metadata.prompt_token_count
        else:  
            prompt_tokens=response.usage.prompt_tokens
            completion_tokens=response.usage.completion_tokens
        total_cost = (prompt_tokens / 1000) * prompt_cost_per_1k + (completion_tokens / 1000) * completion_cost_per_1k
        return total_cost
    
def create_qwen_prompt(instruction, response_structure, input, enable_thinking: bool):
    if enable_thinking:
        qwen_template=f"""<|im_start|>{instruction}
{response_structure}
<|im_end|>
<|im_start|>
{input}
<|im_end|>
<|im_start|>assistant
""" 
    else:  
        qwen_template=f"""<|im_start|>{instruction}
{response_structure}
<|im_end|>
<|im_start|>
{input}
<|im_end|>
<|im_start|>assistant
<think>


</think>
"""
    return qwen_template

def create_normal_prompt(instruction, response_structure, input):
    template=f"""{instruction}
{input}
{response_structure}
"""
    return template

def create_llm_prompt(is_model_qwen,template,reasoning=False):
    instruction= template['instruction']
    response_structure= template['response_structure']
    input= template['input']
    if is_model_qwen:
          return create_qwen_prompt(instruction, response_structure, input, reasoning)
    else:
          return create_normal_prompt(instruction, response_structure, input)