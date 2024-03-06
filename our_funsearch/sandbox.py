from typing import Any
import numpy as np


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
      self,
      program: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    program = program.replace("@funsearch.run", "").replace("@funsearch.evolve", "")
    # print(program)
    # print(test_input)
    # print(timeout_seconds)
    
    try:
      sandbox_namespace = {}
      fun = compile(program,'<string>','exec')
      exec(fun, sandbox_namespace)
      error = sandbox_namespace['main']()
      return [error, True]
    except:
      print("Run Fail!")
      return [np.inf, False]
    # raise NotImplementedError(
    #     'Must provide a sandbox for executing untrusted code.')
