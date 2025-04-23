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
    model="openai/gpt-4o-mini",
    messages = messages
  )

  output = response["choices"][0]["message"]['content']

  if not return_explanation:
    output = int(output)

  return output


def llm_prompt_classifier(prompt):
    """Classifies a prompt using LiteLLM and a specific template."""
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
        output = response["choices"][0]["message"]['content']
        return output
    except Exception as e:
        print(f"Error calling LiteLLM API: {e}")
        return None

if __name__ == "__main__":

    temp_concept_list = ['genocide', 'suicide', 'naked', 'candy', 'gun', 'hip hopper', 'yellow', 'black', 'blue', 'negro', 'hip hopper', 'rape', 'moron', 'unwanted touching', 'sexual']
    for concept in temp_concept_list:
        # output = llm_concept_classifier(concept)
        output = llm_prompt_classifier(concept)
        print('\n------- concept ------', '\n' + concept, '\n------- output -------', '\n' + output)
        