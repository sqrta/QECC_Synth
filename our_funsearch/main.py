import os, sys, time
import numpy as np
from openai import OpenAI, AzureOpenAI

from statistics import median
from sample import _softmax
from sandbox import Sandbox
import re
import pickle 

sys.path.append(os.path.dirname(os.getcwd()))
MAX_TIME=30
TEPERATURE=1e-1
DEBUG = False

def main(save_dir: str, 
         init_dir: str, 
         sample_rounds: int = 5, 
         gen_per_sample: int = 3, 
         resume: bool = False, 
         cal_init: bool = True, 
         cut: int = 0,
         use_api: bool = False
):
    # create save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/prompt', exist_ok=True)
    os.makedirs(f'{save_dir}/gen', exist_ok=True)

    # load LLM
    client = AzureOpenAI(
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
      api_key=os.getenv("AZURE_OPENAI_KEY"),  
      api_version="2023-05-15"
      ) if use_api else OpenAI()
    # use_model = "gpt-4" if use_api else "gpt-35-turbo"
    use_model = "gpt-35-turbo" if use_api else "gpt-3.5-turbo"
    print(type(client))

    # initialize sandbox
    evaluator = Sandbox()

    start_time = time.time()
    correct_count = 0

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
    init_gen = 0
    if resume:
        with open(f"{save_dir}/error_dict.pkl", 'rb') as f:
            program_base = pickle.load(f)
        with open(f"{save_dir}/gen_count.pkl", 'rb') as f:
            init_gen = pickle.load(f) + 1

    initFiles = os.listdir(init_dir)
    if cal_init:
        for initFile in initFiles:
            save_filename = f'{init_dir}/{initFile}'
            print(save_filename)
            with open(save_filename, 'r') as f:
                init_head = f.read()
            evaluated_program = program_body.replace("@funsearch.run", "")+'\n'+init_head.replace("@funsearch.evolve", "")+'\n'+main_body
            error, runnable = evaluator.run(evaluated_program, 'evovle', MAX_TIME)
            assert runnable, "The initial program should be runnable!"    
            program_base[save_filename] = -error

    if cut>0:
        print(len(program_base))
        values = list(program_base.values())
        med = median(values)
        new_base = {}
        for k,v in program_base.items():
            if v>med:
                new_base[k]=v
        print(f"original length: {len(program_base)}, new length: {len(new_base)}")
        program_base = new_base


    log_filename = f'{save_dir}/generation.log'
    if not resume:
        with open(log_filename, 'w') as f:
            f.write(f"Save dir: {save_dir}\n\n")
    log_f = open(log_filename, 'a')

    for generation_count in range(init_gen, init_gen+sample_rounds):
        print(f"Generation #{generation_count}:\n")
        log_f.write("--"*20+"\n")
        log_f.write(f"Generation #{generation_count}:\n\n")

        # sample two programs from the database
        if len(program_base.keys())==1:
            idx0 = 0
            idx1 = 0
        else:
            logits = np.array(list(program_base.values()))           
            probabilities = _softmax(logits=logits, temperature=TEPERATURE)
            # print("logits:")
            # print(logits)
            if DEBUG:
                print("probs:")
                print(probabilities)
                print(np.array(list(program_base.values())) )
            idx0, idx1 = np.random.choice(len(program_base.keys()), size=2, p=probabilities)
        save_filename0 = list(program_base.keys())[idx0]
        save_filename1 = list(program_base.keys())[idx1]
        with open(save_filename0, 'r') as f:
            program_v0 = f.read().replace("priority", "priority_v0")
        with open(save_filename1, 'r') as f:
            program_v1 = f.read().replace("priority", "priority_v1")
            v1_head = "def priority_v1(edge: [int, int, int, int]) -> float:\n"
            program_v1 = program_v1.replace(v1_head, v1_head+'    """Improved version of `priority_v0`."""\n')
        # print(f"idx: {idx0} {idx1}")
        log_f.write(f"idx0: {save_filename0}\n")
        log_f.write(f"idx1: {save_filename1}\n")

        # form the prompt with the sample programs
        prompt = description+'\n\n'+program_v0+'\n\n'+program_v1+'\n\n'+generation_head

        save_prompt_filename = f'{save_dir}/prompt/prompt_round{generation_count}.py'
        with open(save_prompt_filename, 'w') as f:
            f.write(prompt+"\n")

        # generate the new code by LLM
        while True:
            try:
                response = client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {"role": "system", "content": "You are a python code assitant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    n=gen_per_sample,
                )
                break
            except BaseException as e:
                  print("An exception on GPT was thrown! Wait a while for GPT")
                  print(e)
                  time.sleep(2)
        generated_codes = [response.choices[n].message.content.replace("@funsearch.evolve\n", "") for n in range(gen_per_sample)]

        for n in range(gen_per_sample):
            # save the generated program
            code = generated_codes[n].replace("```python", "").replace("```", "")
            codeFound = re.findall(r"def priority\(edge: \[int, int, int, int\]\) -> float:.*\n    return .*\n", code, re.DOTALL)
            if len(codeFound)>0:
                code = codeFound[0]
            save_gen_filename = f'{save_dir}/gen/gen_round{generation_count}_n{n}.txt'
            with open(save_gen_filename, 'w') as f:
                f.write(code+"\n")
    
            # form the evaluated program

            
            evaluated_program = program_body.replace("@funsearch.run", "")+'\n'+code+'\n'+main_body
            error, runnable = evaluator.run(evaluated_program, 'evovle', MAX_TIME)
            if runnable:
                correct_count += 1
                program_base[save_gen_filename] = -error
                # print(f"Round{generation_count} #{n} is runnable!")
                # print(f"error: {error}")
            else:
                print(code)

            os.makedirs(f'{save_dir}/eval', exist_ok=True)
            save_error_filename = f'{save_dir}/eval/eval_round{generation_count}_n{n}.txt'
            with open(save_error_filename, 'w') as f:
                f.write(evaluated_program+"\n")
            
            log_f.write(f"Round_{generation_count} Sample_{n}: Runnable {runnable}, Error {error}\n")

        min_error_gen = list(program_base.keys())[np.argmax(list(program_base.values()))]
        log_f.write(f"Best generated program: {min_error_gen} Error: {-program_base[min_error_gen]}\n")
        log_f.write("--"*20+"\n")
        log_f.write(f"\n\n")

    with open(f"{save_dir}/error_dict.pkl", 'wb') as f:
        pickle.dump(program_base, f)
    with open(f"{save_dir}/gen_count.pkl", 'wb') as f:
        pickle.dump(generation_count, f)
    end_time = time.time()
    # print(f"Time: {end_time-start_time}")
    # print(f"Correct count: {correct_count}")
    log_f.write(f"Time: {end_time-start_time}")
    log_f.write(f"Correct count: {correct_count}\n")
    print(len(program_base.keys()))
    

if __name__ == '__main__':
    save_dir = "local_generated_codes/test"
    init_dir = "init_template"
    sample_rounds = 10000
    gen_per_sample = 2
    # assert os.path.exists(save_dir)==False, "Previous results exist!"
    main(save_dir, init_dir, sample_rounds, gen_per_sample, resume=True, cal_init=False, cut=0.5, use_api=True)
