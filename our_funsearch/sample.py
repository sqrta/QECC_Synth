import numpy as np
import scipy

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  # if not np.all(np.isfinite(logits)):
  #   non_finites = set(logits[~np.isfinite(logits)])
  #   raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result
