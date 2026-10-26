from pathlib import Path
from types import SimpleNamespace
import sys
import time

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]
from transformers import LlamaTokenizer, BertTokenizer
from experiment_data import language_bank
from evaluation_adapter import matched_pool

assets = PHASE / 'results/assets'
llm = LlamaTokenizer.from_pretrained(assets / 'vicuna-7b-v1.3')
llm.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '</s>', 'eos_token': '</s>', 'additional_special_tokens': ['[*]']})
qformer = BertTokenizer.from_pretrained(assets / 'bert-base-uncased')
qformer.add_special_tokens({'bos_token': '[DEC]', 'additional_special_tokens': ['[*]']})
scorer = SimpleNamespace(backbone=SimpleNamespace(llm_tokenizer=llm, tokenizer=qformer))
tick = time.monotonic()
pool = matched_pool(language_bank(), scorer)
print('MATCHED POOL COMPLETE', len(pool.entries), len(pool.pools), time.monotonic() - tick, flush=True)
