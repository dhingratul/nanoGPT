"""
Sample from a trained model
Usage: $ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
"""

from contextlib import nullcontext
import torch
import tiktoken
from model import GPT


def get_model(init_from='gpt2-xl', device='cpu'):
    # -----------------------------------------------------------------------------
    seed = 1337
    if device == "cuda":
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    else:
        dtype = 'float32' # 'float32' or 'bfloat16' or 'float16'
    compile = True
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    print(f"Initializing pre-trained model from {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    return model, ctx


def generate_from_prompt(model, ctx, start="\n", max_new_tokens=10, num_samples=1, device='cpu'):
    # -----------------------------------------------------------------------------
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    # -----------------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    res = []

    # run generation
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                res.append(decode(y[0].tolist()))
    return " ".join(res)

if __name__ == "__main__":
    model, ctx = get_model()
    res = generate_from_prompt(model, ctx, start="What is the answer to life, the universe, and everything?")
    print(res)