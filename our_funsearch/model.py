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
    default_template = 'alpaca'
    print(f"No dialogue template found in model repo. Defaulting to the `{default_template}` template.")
    # dialogue_template = get_dialogue_template("no_system")
    dialogue_template = get_dialogue_template(default_template)

  generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=16,
        max_new_tokens=4096,
    )
  return [tokenizer, model, dialogue_template, generation_config, device]
