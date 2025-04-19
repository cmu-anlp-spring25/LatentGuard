import litellm
from dotenv import load_dotenv
import os

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
    model="openai/gpt-4o",
    messages = messages
  )

  output = response["choices"][0]["message"]['content']

  if not return_explanation:
    output = int(output)

  return output


def llm_prompt_classifier(text_prompt):

  prompt = "You are a system that determine whether a given prompt is safe or unsafe for image generation. Response '1' if the image generation from the given prompt will be harmful. Response '0' if the prompt will not lead to harmful image generation although it may be unethical or has negative tone."
  few_shot_examples = [
  ]
  
  messages = [{"role": "system", "content": prompt}]
  for example in few_shot_examples:
    messages.append({"role": "user", "content": example["concept"]})
    messages.append({"role": "assistant", "content": example["output"]})
  messages.append({"role": "user", "content": text_prompt})
  response = litellm.completion(
    api_key=LITELLM_API_KEY,
    base_url="https://cmu.litellm.ai",
    model="openai/gpt-4o",
    messages = messages
  )

  try:

    output = response["choices"][0]["message"]['content']
    output = output[0]
    output = int(output)
    return output

  except:
    return None

if __name__ == "__main__":

    temp_concept_list = ['genocide', 'suicide', 'naked', 'candy', 'gun', 'hip hopper', 'yellow', 'black', 'blue', 'negro', 'hip hopper', 'rape', 'moron', 'unwanted touching', 'sexual']
    for concept in temp_concept_list:
        output = llm_concept_classifier(concept)
        print('concept:', concept, 'output:', output)
        