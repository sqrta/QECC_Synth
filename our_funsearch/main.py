import os, sys, time
import numpy as np
from openai import OpenAI

from sample import _softmax
from sandbox import Sandbox

sys.path.append(os.path.dirname(os.getcwd()))
MAX_TIME=30
TEPERATURE=1e-3
DEBUG = False


def main(save_dir: str, sample_rounds: int=5, gen_per_sample: int=3):
    # create save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/prompt', exist_ok=True)
    os.makedirs(f'{save_dir}/gen', exist_ok=True)

    # load LLM
    client = OpenAI()

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
    evaluated_program = program_body.replace("@funsearch.run", "")+'\n'+generation_head.replace("@funsearch.evolve", "")+'\n'+main_body
    error, runnable = evaluator.run(evaluated_program, 'evovle', MAX_TIME)
    assert runnable, "The initial program should be runnable!"
    save_filename = f'{save_dir}/initial.py'
    with open(save_filename, 'w') as f:
        f.write(generation_head.replace("@funsearch.evolve\n", ""))
    program_base[save_filename] = -error

    log_filename = f'{save_dir}/generation.log'
    with open(log_filename, 'w') as f:
        f.write(f"Save dir: {save_dir}\n\n")
    log_f = open(log_filename, 'a')

    for generation_count in range(sample_rounds):
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
        prompt = description+program_body+'\n\n'+program_v0+'\n\n'+program_v1+'\n\n'+generation_head

        save_prompt_filename = f'{save_dir}/prompt/prompt_round{generation_count}.py'
        with open(save_prompt_filename, 'w') as f:
            f.write(prompt+"\n")

        # generate the new code by LLM  
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a python code assitant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            n=gen_per_sample,
        )
        generated_codes = [response.choices[n].message.content.replace("@funsearch.evolve\n", "") for n in range(gen_per_sample)]

        for n in range(gen_per_sample):
            # save the generated program
            code = generated_codes[n].replace("```python", "").replace("```", "")
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
            # else:
            
            os.makedirs(f'{save_dir}/eval', exist_ok=True)
            save_error_filename = f'{save_dir}/eval/eval_round{generation_count}_n{n}.txt'
            with open(save_error_filename, 'w') as f:
                f.write(evaluated_program+"\n")
            
            log_f.write(f"Round_{generation_count} Sample_{n}: Runnable {runnable}, Error {error}\n")

        min_error_gen = list(program_base.keys())[np.argmax(list(program_base.values()))]
        log_f.write(f"Best generated program: {min_error_gen} Error: {-program_base[min_error_gen]}\n")
        log_f.write("--"*20+"\n")
        log_f.write(f"\n\n")


    end_time = time.time()
    # print(f"Time: {end_time-start_time}")
    # print(f"Correct count: {correct_count}")
    log_f.write(f"Time: {end_time-start_time}")
    log_f.write(f"Correct count: {correct_count}")
    

if __name__ == '__main__':
    save_dir = "local_generated_codes/test"
    sample_rounds = 5
    gen_per_sample = 1
    # assert os.path.exists(save_dir)==False, "Previous results exist!"
    main(save_dir, sample_rounds, gen_per_sample)
