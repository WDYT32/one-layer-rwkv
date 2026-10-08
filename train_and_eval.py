import argparse
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_transformer import BaselineTransformer
from model_rwkv7 import RWKV7Model


class Vocab:
    def __init__(self):
        chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '+', '-', '*', '(', ')', '=', '_', ':', ';', '<eos>', '<pad>']
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.pad_id = self.stoi['<pad>']
        self.eos_id = self.stoi['<eos>']
        self.eq_id = self.stoi['=']
        self.fill_id = self.stoi['_']
        self.colon_id = self.stoi[':']   # hybrid/compact formats: 'redex : value'
        self.semi_id = self.stoi[';']    # hybrid format: 'value ; rewritten expression'
        self.vocab_size = len(chars)

    def encode(self, string):
        # KeyError on an unknown token instead of silently dropping it
        return [self.stoi[t] for t in string.strip().split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)


def answer_start(ids, vocab):
    """Index where the final answer begins: just after the last '=', ':' or ';'
    (trace: after the last '='; hybrid: after the last ';'; compact: after the last ':')."""
    marks = {vocab.eq_id, vocab.colon_id, vocab.semi_id}
    return max(i for i, t in enumerate(ids) if t in marks) + 1


def apply_fillers(text, mode):
    """Filler ('_') variants. The solution-trace data has no fillers, so every
    mode is a no-op on it; the function is kept for old filler-style data.
    Only the part before the FIRST '=' counts as the input expression."""
    toks = text.split()
    eq = toks.index('=')
    left, right = toks[:eq], toks[eq + 1:]
    if mode in ('none', 'answer'):
        left = [t for t in left if t != '_']
    if mode in ('none', 'input'):
        right = [t for t in right if t != '_']
    return " ".join(left + ['='] + right)


class MathDataset(Dataset):
    def __init__(self, filepath, vocab, max_len=320, fillers='both'):
        self.items = []
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                ids = vocab.encode(apply_fillers(item['text'], fillers))
                if len(ids) > max_len:
                    raise ValueError(f"sequence of {len(ids)} tokens exceeds max_len={max_len}; "
                                     f"raise --max_len (and the model's context size, if it has one)")
                self.items.append((ids, item['level']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collate(pad_id):
    def collate(batch):
        seqs, levels = zip(*batch)
        L = max(len(s) for s in seqs)  # pad to the longest sequence in the batch
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out, torch.tensor(levels, dtype=torch.long)
    return collate


class LengthBucketSampler(Sampler):
    """Batch sampler that puts sequences of similar length together, so padding
    (and with it the time-recurrence length of RWKV) is no longer set by the one
    longest example in a random batch of 64.
    shuffle=True : shuffle, cut into pools of `pool_batches` batches, sort each pool
                   by length, cut into batches, shuffle the batches (new order per epoch).
    shuffle=False: one global sort by length (evaluation; order does not matter).
    Side effect to keep in mind: batches become nearly single-level (length tracks the
    number of operators), so gradient noise differs from fully random batches."""

    def __init__(self, lengths, batch_size, shuffle=True, pool_batches=50, seed=0):
        self.lengths, self.bs, self.shuffle, self.seed = lengths, batch_size, shuffle, seed
        self.pool = pool_batches * batch_size if shuffle else max(len(lengths), 1)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.pool):
            pool = sorted(idx[s:s + self.pool], key=lambda i: self.lengths[i])
            batches += [pool[i:i + self.bs] for i in range(0, len(pool), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n, full = len(self.lengths), len(self.lengths) // self.pool
        per_pool = -(-self.pool // self.bs)
        return full * per_pool + -(-(n - full * self.pool) // self.bs)


def padded_fraction(batches, lengths):
    """Share of tokens in the padded batches that are real (1.0 = no padding)."""
    real = padded = 0
    for b in batches:
        ls = [lengths[i] for i in b]
        real += sum(ls)
        padded += max(ls) * len(ls)
    return real / max(padded, 1)


def get_masks(batch, vocab):
    """Masks over targets = batch[:, 1:].
    Sequence layout: <expression> = <step 1> = <step 2> = ... = <answer> <eos>
    The input expression is random operands, so it is never a target by default.
    post   : everything after the FIRST '=' (whole solution trace + <eos>)
    answer : post without filler tokens (for trace data identical to post)"""
    targets = batch[:, 1:]
    eq_pos = (batch == vocab.eq_id).float().argmax(dim=1)  # index of the first '=' in batch
    idx = torch.arange(targets.size(1), device=batch.device).unsqueeze(0)
    after_eq = idx >= eq_pos.unsqueeze(1)  # targets[j] = batch[j+1] is after '=' iff j >= eq_pos
    valid = targets != vocab.pad_id
    post = after_eq & valid
    answer = post & (targets != vocab.fill_id)
    return targets, valid, post, answer


def masked_ce(logits, targets, mask, vocab_size):
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1),
                         reduction='none').view_as(targets)
    return (ce * mask).sum() / mask.sum().clamp(min=1), ce


def step_correctness(hit, batch, solution, vocab):
    """Per-step correctness under teacher forcing (gold prefix at every position).
    Step k (k = 1..level) is the k-th segment after the first '='; it includes the
    '=' (or <eos>) that closes it, so 'the model knows the step is finished' is
    part of being correct. Returns (n_steps, n_ok), both (B,) long tensors."""
    seg = (batch[:, :-1] == vocab.eq_id).long().cumsum(1)  # '=' count among the inputs
    seg = seg * solution                                    # 0 outside the solution
    K = int(seg.max().item())
    n_steps = torch.zeros(batch.size(0), dtype=torch.long, device=batch.device)
    n_ok = torch.zeros_like(n_steps)
    for k in range(1, K + 1):
        sel = seg == k
        present = sel.any(1)
        ok = (hit | ~sel).all(1) & present
        n_steps += present.long()
        n_ok += ok.long()
    return n_steps, n_ok


@torch.no_grad()
def evaluate(model, loader, vocab, device):
    """Teacher-forced evaluation on the whole split. Only the solution (everything
    after the first '=') is scored.
    em       : every solution token is the argmax given the gold prefix. By induction
               this is identical to greedy generation reproducing the whole gold
               trace, so it is the 'whole chain correct' rate (a lower bound on
               final-answer accuracy; see generate_eval for the real one).
    step_acc : fraction of steps (one operation each) predicted fully correctly
               given the gold previous steps = per-operation reliability."""
    model.eval()
    ce_sum, tok_cnt, tok_ok = 0.0, 0, 0
    stats = defaultdict(lambda: {'n': 0, 'em': 0, 'steps': 0, 'steps_ok': 0})

    for batch, levels in loader:
        batch, levels = batch.to(device), levels.to(device)
        logits = model(batch[:, :-1])
        targets, valid, post, answer = get_masks(batch, vocab)
        _, ce = masked_ce(logits, targets, answer, vocab.vocab_size)
        hit = logits.argmax(-1) == targets

        ce_sum += (ce * answer).sum().item()
        tok_cnt += answer.sum().item()
        tok_ok += (hit & answer).sum().item()

        ok = (hit | ~answer).all(dim=1)
        n_steps, n_ok = step_correctness(hit, batch, answer, vocab)
        for lvl in levels.unique().tolist():
            sel = levels == lvl
            s = stats[lvl]
            s['n'] += sel.sum().item()
            s['em'] += ok[sel].sum().item()
            s['steps'] += n_steps[sel].sum().item()
            s['steps_ok'] += n_ok[sel].sum().item()

    n_total = sum(s['n'] for s in stats.values())
    steps_total = sum(s['steps'] for s in stats.values())
    return {
        'answer_loss': ce_sum / max(tok_cnt, 1),
        'answer_token_acc': tok_ok / max(tok_cnt, 1),
        'em': sum(s['em'] for s in stats.values()) / max(n_total, 1),
        'step_acc': sum(s['steps_ok'] for s in stats.values()) / max(steps_total, 1),
        'by_level': {int(l): {'n': s['n'], 'em': s['em'] / s['n'],
                              'step_acc': s['steps_ok'] / max(s['steps'], 1)}
                     for l, s in sorted(stats.items())},
    }


@torch.no_grad()
def generate_eval(model, dataset, vocab, device, max_len, n=1000, batch_size=128):
    """Free-running greedy generation: the model gets '<expression> =' and writes
    the whole trace itself; we score only the number after the LAST '='.
    Unlike teacher-forced EM this lets a chain contain a wrong step as long as
    the final number comes out right, and counts a missing <eos> as wrong.

    No KV/state cache: every new token re-runs the model on the whole prefix, so
    the cost is O(T^2) model-time per batch. That is why it is run on a subset
    and, by default, only after the last epoch. Prompts are grouped by length so
    no padding enters the model. Note: a transformer with a fixed learned-position
    table must support max_len positions."""
    model.eval()
    items = dataset.items[:n]
    groups = defaultdict(list)  # prompt length -> [(prompt, gold_final, level)]
    for ids, level in items:
        first_eq = ids.index(vocab.eq_id)
        gold_final = ids[answer_start(ids, vocab):-1]  # drop <eos>
        groups[first_eq + 1].append((ids[:first_eq + 1], gold_final, level))

    stats = defaultdict(lambda: {'n': 0, 'ok': 0, 'no_eos': 0})
    for plen, group in groups.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            seq = torch.tensor([p for p, _, _ in chunk], dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _ in range(max_len - plen):
                nxt = model(seq)[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
                seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
                done |= nxt == vocab.eos_id
                if done.all():
                    break
            for row, (_, gold_final, level) in zip(seq.tolist(), chunk):
                s = stats[level]
                s['n'] += 1
                if vocab.eos_id not in row:
                    s['no_eos'] += 1
                    continue
                row = row[:row.index(vocab.eos_id)]
                s['ok'] += row[answer_start(row, vocab):] == gold_final

    n_total = sum(s['n'] for s in stats.values())
    return {
        'final_em': sum(s['ok'] for s in stats.values()) / max(n_total, 1),
        'no_eos': sum(s['no_eos'] for s in stats.values()) / max(n_total, 1),
        'n': n_total,
        'by_level': {int(l): {'n': s['n'], 'final_em': s['ok'] / s['n']}
                     for l, s in sorted(stats.items())},
    }


def fmt(name, m):
    lv = " ".join(f"L{l}:{v['em']:.3f}" for l, v in m['by_level'].items())
    st = " ".join(f"L{l}:{v['step_acc']:.3f}" for l, v in m['by_level'].items())
    return (f"  {name:9s} loss={m['answer_loss']:.4f} tok_acc={m['answer_token_acc']:.4f} "
            f"chain_EM={m['em']:.3f} step_acc={m['step_acc']:.4f}\n"
            f"            chain_EM by ops | {lv}\n"
            f"            step_acc by ops | {st}")


def fmt_gen(name, g):
    lv = " ".join(f"L{l}:{v['final_em']:.3f}" for l, v in g['by_level'].items())
    return (f"  {name:9s} final_answer_EM(greedy)={g['final_em']:.3f} no_eos={g['no_eos']:.3f} "
            f"(n={g['n']}) | {lv}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['transformer', 'rwkv', 'rwkv-2x'])
    parser.add_argument('--pos', type=str, default='learned', choices=['learned', 'rope'],
                        help='positional scheme of the transformer')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--max_len', type=int, default=320,
                        help='longest allowed sequence. Solution traces reach ~250 tokens at 9 '
                             'operators; a transformer with learned positions must be built for this many')
    parser.add_argument('--loss_on', type=str, default='answer', choices=['answer', 'after_eq', 'all'],
                        help="'answer' (default): loss on the whole solution trace (everything after the "
                             "first '='); 'after_eq': same, plus filler tokens on old filler-style data; "
                             "'all': also on the random operands of the input expression")
    parser.add_argument('--fillers', type=str, default='both',
                        choices=['both', 'input', 'answer', 'none'],
                        help="filler variants for OLD filler-style data; no effect on solution-trace data")
    parser.add_argument('--threads', type=int, default=None,
                        help='torch.set_num_threads(n); default: PyTorch default (usually the physical cores)')
    parser.add_argument('--no_bucket', action='store_true',
                        help='disable length bucketing (random batches padded to their longest sequence)')
    parser.add_argument('--gen_eval', action='store_true',
                        help='after the last epoch also run free-running greedy generation and score '
                             'the final answer (slow: O(T^2) model calls, no cache)')
    parser.add_argument('--gen_n', type=int, default=1000, help='samples per split for --gen_eval')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads: {torch.get_num_threads()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocab()
    collate = make_collate(vocab.pad_id)

    train_ds = MathDataset(f'{args.data_dir}/train.jsonl', vocab, args.max_len, args.fillers)
    test_id_ds = MathDataset(f'{args.data_dir}/test_id.jsonl', vocab, args.max_len, args.fillers)
    def make_loader(ds, shuffle):
        if args.no_bucket:
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate), None
        lengths = [len(ids) for ids, _ in ds.items]
        sampler = LengthBucketSampler(lengths, args.batch_size, shuffle, seed=args.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate), sampler

    train_loader, train_sampler = make_loader(train_ds, True)
    id_loader, _ = make_loader(test_id_ds, False)
    ood_path = f'{args.data_dir}/test_ood.jsonl'
    ood_loader, test_ood_ds = None, None  # exists only if train covers fewer operators than the generator's max
    if os.path.exists(ood_path):
        test_ood_ds = MathDataset(ood_path, vocab, args.max_len, args.fillers)
        ood_loader, _ = make_loader(test_ood_ds, False)

    if args.model == 'transformer':
        model = BaselineTransformer(vocab.vocab_size, args.d_model, args.n_head,
                                    args.n_layer, pos=args.pos).to(device)
        name = f"transformer-{args.pos}-L{args.n_layer}-f{args.fillers}-s{args.seed}{args.tag}"
    elif args.model == 'rwkv-2x':
        # 1 шар, але з подвійними параметрами (через dim_att та ffn_expand=12) і вдвічі більшою кількістю голів
        model = RWKV7Model(vocab.vocab_size, args.d_model, n_head=args.n_head * 2, n_layer=1, dim_att=args.d_model * 2, ffn_expand=12).to(device)
        name = f"rwkv-2x-L1-f{args.fillers}-s{args.seed}{args.tag}"
    else:
        model = RWKV7Model(vocab.vocab_size, args.d_model, args.n_head, args.n_layer).to(device)
        name = f"rwkv-channelmixing-L{args.n_layer}-f{args.fillers}-s{args.seed}{args.tag}"

    n_params = count_parameters(model)
    print(f"Run: {name}\nParameters: {n_params:,}")
    if train_sampler is not None:
        lens = [len(ids) for ids, _ in train_ds.items]
        rnd = random.Random(0)
        shuf = list(range(len(lens)))
        rnd.shuffle(shuf)
        plain = [shuf[i:i + args.batch_size] for i in range(0, len(shuf), args.batch_size)]
        print(f"Length bucketing: {padded_fraction(plain, lens):.0%} -> "
              f"{padded_fraction(list(train_sampler), lens):.0%} of batch tokens are real (rest is padding)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer = SummaryWriter(log_dir=f'runs/{name}')
    history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch, _ in pbar:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets, valid, post, answer = get_masks(batch, vocab)
            mask = {'answer': answer, 'after_eq': post, 'all': valid}[args.loss_on]

            optimizer.zero_grad()
            logits = model(inputs)
            loss, _ = masked_ce(logits, targets, mask, vocab.vocab_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", T=batch.size(1))
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        splits = {'id': evaluate(model, id_loader, vocab, device)}
        if ood_loader is not None:
            splits['ood'] = evaluate(model, ood_loader, vocab, device)
        print(f"Epoch {epoch + 1}")
        for split, m in splits.items():
            print(fmt(f'test_{split}', m))
            writer.add_scalar(f'AnswerLoss/{split}', m['answer_loss'], epoch)
            writer.add_scalar(f'AnswerTokenAcc/{split}', m['answer_token_acc'], epoch)
            writer.add_scalar(f'EM/{split}', m['em'], epoch)
            writer.add_scalar(f'StepAcc/{split}', m['step_acc'], epoch)
            for lvl, v in m['by_level'].items():
                writer.add_scalar(f'EM_ops_{lvl}', v['em'], epoch)
                writer.add_scalar(f'StepAcc_ops_{lvl}', v['step_acc'], epoch)
        entry = {'epoch': epoch + 1, **{f'test_{k}': v for k, v in splits.items()}}

        if args.gen_eval and epoch + 1 == args.epochs:
            gens = {'id': generate_eval(model, test_id_ds, vocab, device, args.max_len, args.gen_n)}
            if test_ood_ds is not None:
                gens['ood'] = generate_eval(model, test_ood_ds, vocab, device, args.max_len, args.gen_n)
            for split, g in gens.items():
                print(fmt_gen(f'test_{split}', g))
                writer.add_scalar(f'FinalAnswerEM/{split}', g['final_em'], epoch)
                entry[f'gen_{split}'] = g
        history.append(entry)

    os.makedirs('results', exist_ok=True)
    with open(f'results/{name}.json', 'w') as f:
        json.dump({'args': vars(args), 'params': n_params, 'history': history}, f, indent=2)
    print(f"Saved results/{name}.json")


if __name__ == "__main__":
    train()
