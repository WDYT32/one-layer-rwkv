import sys
sys.path.append('.')
from model_rwkv7 import RWKV7Model
from train_and_eval import count_parameters

class FakeVocab:
    vocab_size = 21

vocab = FakeVocab()

base = RWKV7Model(vocab.vocab_size, d_model=256, n_head=4, n_layer=1, dim_att=None, ffn_expand=4)
print(f"Base: {count_parameters(base)}")

x2 = RWKV7Model(vocab.vocab_size, d_model=256, n_head=8, n_layer=1, dim_att=512, ffn_expand=12)
print(f"2x: {count_parameters(x2)}")

# let's try some values for x4
for ffn in [16, 20, 24, 28]:
    x4 = RWKV7Model(vocab.vocab_size, d_model=256, n_head=16, n_layer=1, dim_att=1024, ffn_expand=ffn)
    print(f"x4 with ffn={ffn}: {count_parameters(x4)}")

