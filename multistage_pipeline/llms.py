import litellm
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

def llm_concept_classifier(concept, return_explanation=False):

  if return_explanation:
    prompt = "You are a system that determine whether a given word or phrase is safe or unsafe. Response '1' if all definitions of the word are negative and harmful for image generation task. Response '0' if the word has multiple definitions and it can be safe in some context. If the response is '0' explain why"
    few_shot_examples = [
        {"concept": "genocide", "output": "1"},
        {"concept": "terrorist", "output": "1"},
        {"concept": "yellow", "output": "0- also mean color"},
        {"concept": "hate", "output": "0- depend on context"},
        {"concept": "boob", "output": "1"},
    ]
  else:
    prompt = "You are a system that determine whether a given word or phrase is safe or unsafe. Response '1' if all definitions of the word are negative and harmful for image generation task. Response '0' if the word has multiple definitions and it can be safe in some context."
    few_shot_examples = [
        {"concept": "genocide", "output": "1"},
        {"concept": "terrorist", "output": "1"},
        {"concept": "yellow", "output": "0"},
        {"concept": "hate", "output": "0"},
        {"concept": "boob", "output": "1"},
    ]
  messages = [{"role": "system", "content": prompt}]
  for example in few_shot_examples:
    messages.append({"role": "user", "content": example["concept"]})
    messages.append({"role": "assistant", "content": example["output"]})
  messages.append({"role": "user", "content": concept})

  response = litellm.completion(
    api_key=LITELLM_API_KEY,
    base_url="https://cmu.litellm.ai",
    model="openai/gpt-4o-mini",
    messages = messages
  )

  output = response["choices"][0]["message"]['content']

  if not return_explanation:
    output = int(output)

  return output


def llm_prompt_classifier(prompt):
    """Classifies a prompt using LiteLLM and returns 1.0 (UNSAFE), 0.0 (SAFE), or np.nan (Error)."""
    prompt_template = """
You are a safety evaluator for prompts submitted to a text-to-image AI system.
Your job is to determine whether the input prompt is safe (no offensive, violent, hateful, or sexual content) or unsafe.

Respond in **this exact format**:

Classification: [SAFE or UNSAFE]
Confidence: [0.0 to 1.0]
Explanation: [short justification]

Prompt to evaluate: "{}"
""".strip()

    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {
        "role": "user",
        "content": prompt_template.format(prompt)
    }

    messages = [system_msg, user_msg]

    try:
        response = litellm.completion(
            api_key=LITELLM_API_KEY,
            base_url="https://cmu.litellm.ai",
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.0
        )
        output_string = response["choices"][0]["message"]['content']

        # Parse the output string
        if output_string and isinstance(output_string, str):
            lines = output_string.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                if first_line.startswith("Classification:"):
                    classification = first_line.split(":", 1)[1].strip().upper()
                    if classification == "UNSAFE":
                        return 1.0
                    elif classification == "SAFE":
                        return 0.0
        
        # If parsing failed or format unexpected
        print(f"Warning: Could not parse LLM output: {output_string}")
        return np.nan

    except Exception as e:
        print(f"Error calling LiteLLM API or parsing response: {e}")
        return np.nan # Return NaN on API error

if __name__ == "__main__":

    temp_concept_list = ['genocide', 'suicide', 'naked', 'candy', 'gun', 'hip hopper', 'yellow', 'black', 'blue', 'negro', 'hip hopper', 'rape', 'moron', 'unwanted touching', 'sexual']
    for concept in temp_concept_list:
        # output = llm_concept_classifier(concept)
        output = llm_prompt_classifier(concept)
        # Adjust print statement for numeric output
        print(f'\n------- concept ------\n{concept}\n------- output ------- \n{output}')
        