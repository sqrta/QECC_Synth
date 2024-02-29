import torch
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)


def _initialize_real_llm(checkpoint: str, cache_dir: str) -> list:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=None, cache_dir=cache_dir)
  model = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=None, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir
  )
  try:
    dialogue_template = DialogueTemplate.from_pretrained(checkpoint, revision=None)
  except Exception:
    print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
    # dialogue_template = get_dialogue_template("no_system")
    dialogue_template = get_dialogue_template("alpaca")

  generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
    )
  return [tokenizer, model, dialogue_template, generation_config, device]


def main():
  checkpoint = 'bigcode/starcoder'
  cache_dir = '/cmlscratch/dengch/code_llm/starcoder/cache'
  tokenizer, model, dialogue_template, generation_config, device = _initialize_real_llm(checkpoint, cache_dir)

  with open('capset1.py', 'r') as f:
    code = f.read()
  description = "Modifiy the python function 'priority' in the following code block to maximize the returned value of python function 'evaluate'. The answer should be only the code for the new python function whose name is 'priority'. And the answer should be not commented out."

  for n in range(2):
    print(f"Round {n}:")
    prompt_with_role = {"role": "user", "content": description+code+'\n'}
    dialogue_template.messages = [prompt_with_role]
    formatted_prompt = dialogue_template.get_inference_prompt()
    print("<The prompt begins>")
    print(formatted_prompt)
    print("<The prompt ends>\n")

    # inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    outputs = model.generate(**inputs, generation_config=generation_config)
    #generated_code = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=False).lstrip()

    repsponse_token = "### Response:\n"
    loc0 = generated_code.find(repsponse_token)
    end_token = "<|endoftext|>"
    loc1 = generated_code.find(end_token)
    generated_code = generated_code[loc0+len(repsponse_token):loc1]
    print("<The generation begins>")
    print(generated_code)
    print("<The generation ends>\n\n")

if __name__ == '__main__':
  main()