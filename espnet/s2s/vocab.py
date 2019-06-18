import re

from . import phones_en

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def sequence_to_phoneme(seqs, language):
  if language == "en":
    phoneme_to_id = phones_en.phone_to_id
  else:
    raise ValueError("Unsupported language! from function text/phoneme_to_sequence")
  id_to_phoneme = {}
  for k, v in phoneme_to_id.items():
    id_to_phoneme[v] = k
  sequence = [id_to_phoneme[w] for w in seqs if w != 0]
  return sequence

def phoneme_to_sequence(phones, language="en"):
  if language == "en":
    phoneme_to_id = phones_en.phone_to_id
  else:
    raise ValueError("Unsupported language! from function text/phoneme_to_sequence")

  sequence = [phoneme_to_id[phone] for phone in phones.split(' ') if phone in phoneme_to_id]
  return sequence



