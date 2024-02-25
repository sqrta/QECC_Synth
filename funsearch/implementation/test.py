from funsearch import _extract_function_names, main
from config import Config

with open('capset.py', 'r') as f:
    content = f.read()
    res = _extract_function_names(content)
    conf = Config()
    inputs = [3]
    print(res)
    checkpoint = 'bigcode/starcoder'
    cache_dir = 'your path to save llm checkpoint'
    main(content, inputs, conf, checkpoint, cache_dir)
    