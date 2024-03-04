import torch
import os, sys
import numpy as np
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)


from model import _initialize_real_llm
from sample import _softmax
from sandbox import Sandbox


sys.path.append(os.path.dirname(os.getcwd()))
MAX_TIME=30
TEPERATURE=0.1

def main(model_cache_dir: str, save_dir: str):
  # create save_dir
  os.makedirs(save_dir, exist_ok=True)

  # load LLM
  checkpoint = 'bigcode/starcoder'
  tokenizer, model, dialogue_template, generation_config, device = _initialize_real_llm(checkpoint, model_cache_dir)

  # initialize sandbox
  evaluator = Sandbox()

  # initialize template resource
  with open('template_source/description.txt', 'r') as f:
    description = f.read()
  with open('template_source/program_body.py', 'r') as f:
    program_body = f.read()
  with open('template_source/generation_head.py', 'r') as f:
    generation_head = f.read()
  with open('template_source/main_body.py', 'r') as f:
    main_body = f.read()

  # initilize the program database
  program_base = {}
  evaluated_program = program_body.replace("@funsearch.run", "")+'\n'+generation_head.replace("@funsearch.evolve", "")+'\n'+main_body
  error, runnable = evaluator.run(evaluated_program, 'evovle', MAX_TIME)
  assert runnable, "The initial program should be runnable!"
  save_filename = f'{save_dir}/initial.py'
  with open(save_filename, 'w') as f:
    f.write(generation_head.replace("@funsearch.evolve\n", ""))
  program_base[save_filename] = error

  generation_n = 10
  for generation_count in range(generation_n):
    print(f"Generation #{generation_count}:")

    # sample two programs from the database
    if len(program_base.keys())==1:
        save_filename = list(program_base.keys())[0]
        with open(save_filename, 'r') as f:
            program = f.read()
        program_v0 = program.replace("priority", "priority_v0")
        program_v1 = program.replace("priority", "priority_v1")
    else:
        probabilities = _softmax(logits=np.ndarray(list(program_base.values)), temperature=TEPERATURE)
        idx0, idx1 = np.random.choice(len(program_base.keys()), size=2, p=probabilities)
        save_filename0 = program_base.keys()[idx0]
        save_filename1 = program_base.keys()[idx1]
        with open(save_filename0, 'r') as f:
            program_v0 = f.read().replace("priority", "priority_v0")
        with open(save_filename1, 'r') as f:
            program_v1 = f.read().replace("priority", "priority_v1")

    # form the prompt with the sample programs
    prompt = description+program_body+'\n\n'+program_v0+'\n\n'+program_v1+'\n\n'+generation_head

    # generate the new code by LLM
    prompt_with_role = {"role": "user", "content": prompt+'\n'}
    dialogue_template.messages = [prompt_with_role]
    formatted_prompt = dialogue_template.get_inference_prompt()
    inputs = tokenizer(formatted_prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    outputs = model.generate(**inputs, generation_config=generation_config)
    ori_generated_code = tokenizer.decode(outputs[0], skip_special_tokens=False).lstrip()

    # postprocessing
    response_token = "### Response:"
    loc0 = ori_generated_code.find(response_token)
    end_token = "<|endoftext|>"
    loc1 = ori_generated_code.find(end_token)
    generated_code = ori_generated_code[loc0+len(response_token):loc1] if loc1!=-1 else ori_generated_code[loc0+len(response_token):]

    # form the evaluated program
    evaluated_program = program_body.replace("@funsearch.run", "")+'\n'+generated_code+'\n'+main_body
    error, runnable = evaluator.run(evaluated_program, 'evovle', MAX_TIME)

    # write the runnable program
    if runnable:
      save_filename = f'{save_dir}/generation_round{generation_count}.py'
      with open(save_filename, 'w') as f:
        f.write(generated_code+"\n")
      program_base[save_filename] = error


if __name__ == '__main__':
  model_cache_dir = '/cmlscratch/dengch/code_llm/starcoder/cache'
  save_dir = "generated_codes"
  main(model_cache_dir, save_dir)