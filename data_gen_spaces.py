import argparse
import json
import os
import random
from collections import Counter, defaultdict

OPS = ['+', '-', '*']


def apply_op(op, a, b):
    return a + b if op == '+' else a - b if op == '-' else a * b


def build(k, rng, max_abs, is_root=True):
    """Random full binary expression tree with exactly k operators.
    Every non-leaf child is parenthesised (no precedence needed).
    The operator at each node is picked among those that keep the node's value
    within +-max_abs, so *every* intermediate result stays bounded. A valid
    operator always exists: min(|a+b|, |a-b|) = ||a|-|b|| <= max(|a|, |b|)."""
    if k == 0:
        d = rng.randint(1, 9)
        return str(d), d
    left_ops = rng.randint(0, k - 1)
    ls, lv = build(left_ops, rng, max_abs, False)
    rs, rv = build(k - 1 - left_ops, rng, max_abs, False)
    valid = [o for o in OPS if abs(apply_op(o, lv, rv)) <= max_abs]
    op = rng.choice(valid)
    s = f"{ls}{op}{rs}"
    return (s if is_root else f"({s})"), apply_op(op, lv, rv)


def generate_expression(level, rng, max_abs, exclude=None):
    """level = number of operators. Level 1 has only 243 distinct expressions,
    so it is never excluded (that would loop forever)."""
    while True:
        expr, val = build(level, rng, max_abs)
        if exclude is not None and level >= 2 and expr in exclude:
            continue
        return expr, val


def format_expr(expr_str, result_val):
    """A filler '_' after every input symbol and 2 fillers per operator between
    '=' and the answer (variants are derived from this in train_and_eval.py)."""
    num_ops = sum(c in '+-*' for c in expr_str)
    left_part = " _ ".join(list(expr_str))
    time_tokens = " ".join(["_"] * (num_ops * 2))
    res_str = str(result_val)
    if res_str.startswith('-'):
        res_part = "- " + " ".join(list(res_str[1:]))
    else:
        res_part = " ".join(list(res_str))

    formatted = f"{left_part} _ = "
    if num_ops > 0:
        formatted += f"{time_tokens} "
    formatted += f"{res_part} <eos>"
    return formatted


def generate_dataset(num_samples, levels, filepath, rng, max_abs, exclude=None):
    data, exprs, max_len = [], set(), 0
    answers = defaultdict(list)
    for _ in range(num_samples):
        level = rng.choice(levels)
        expr_str, val = generate_expression(level, rng, max_abs, exclude)
        text = format_expr(expr_str, val)
        data.append({"text": text, "level": level, "expr": expr_str, "answer": val})
        exprs.add(expr_str)
        answers[level].append(val)
        max_len = max(max_len, len(text.split()))

    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  {filepath}: {len(data)} samples, max {max_len} tokens (with all fillers)")
    print("    ops      n   const-guess EM")
    for lvl in sorted(answers):
        c = Counter(answers[lvl])
        print(f"    {lvl:3d} {len(answers[lvl]):6d}   {c.most_common(1)[0][1] / len(answers[lvl]):.3f}")
    return exprs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='data')
    parser.add_argument('--max_ops', type=int, default=9, help='largest number of operators')
    parser.add_argument('--train_max_ops', type=int, default=None,
                        help='train on 1..train_max_ops operators (default: max_ops). If smaller than '
                             'max_ops, test_ood.jsonl holds the longer expressions.')
    parser.add_argument('--max_abs', type=int, default=999, help='bound on every intermediate value')
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_test', type=int, default=9000)
    args = parser.parse_args()

    train_max = args.train_max_ops or args.max_ops
    assert 1 <= train_max <= args.max_ops
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    id_levels = list(range(1, train_max + 1))

    print(f"Generating train (1..{train_max} operators)...")
    train_exprs = generate_dataset(args.n_train, id_levels, f'{args.out_dir}/train.jsonl',
                                   rng, args.max_abs)
    print("Generating test_id (same levels, expressions unseen in train for >= 2 operators)...")
    generate_dataset(args.n_test, id_levels, f'{args.out_dir}/test_id.jsonl',
                     rng, args.max_abs, exclude=train_exprs)

    ood_path = f'{args.out_dir}/test_ood.jsonl'
    if train_max < args.max_ops:
        ood_levels = list(range(train_max + 1, args.max_ops + 1))
        print(f"Generating test_ood ({ood_levels[0]}..{ood_levels[-1]} operators, longer than any train expression)...")
        generate_dataset(args.n_test, ood_levels, ood_path, rng, args.max_abs)
    elif os.path.exists(ood_path):
        os.remove(ood_path)  # stale file from an earlier split
    print("Done!")
