import random, json

def sample():
    a, b = random.randint(10, 99), random.randint(10, 99)
    return a, b, random.choice(["+", "-"])

def ans(a, b, op):
    return a + b if op == "+" else a - b

# exact test set from baseline.py (seed 42): never train on it
random.seed(42)
test = {sample() for _ in range(200)}

def wrong_of(a, b, op):
    t = ans(a, b, op)
    while True:
        if op == "+":
            w = t + random.choice([-10, 10, -1, 1, 11, -11])
        else:
            w = random.choice([-t, t + random.choice([-10, 10]), t + random.choice([-1, 1])])
        if w != t:
            return w

def check(a, b, op, r):
    return f"{r} - {b} = {r - b}" if op == "+" else f"{r} + {b} = {r + b}"

def trace(a, b, op, slip):
    t = ans(a, b, op)
    lines = []
    if slip:
        w = wrong_of(a, b, op)
        lines.append(f"{a} {op} {b} = {w}.")
        lines.append(f"Check: {check(a, b, op, w)}, but it should be {a}. "
                     "That doesn't match. Wait, let me redo it.")
        lines.append(f"Redo: {a} {op} {b} = {t}.")
    else:
        lines.append(f"{a} {op} {b} = {t}.")
    lines.append(f"Check: {check(a, b, op, t)}. It matches.")
    lines.append(f"Answer: {t}")
    return "\n".join(lines)

random.seed(7)
seen, rows = set(), []
while len(rows) < 3000:
    a, b, op = sample()
    if (a, b, op) in test or (a, b, op) in seen:
        continue
    seen.add((a, b, op))
    slip = random.random() < 0.35
    prompt = f"What is {a} {op} {b}? Think step by step, then give the final answer after 'Answer:'."
    rows.append({"messages": [{"role": "user", "content": prompt},
                              {"role": "assistant", "content": trace(a, b, op, slip)}]})

with open("train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(len(rows), "examples written")
print(rows[0]["messages"][1]["content"])
print("---")
print(next(r for r in rows if "Wait" in r["messages"][1]["content"])["messages"][1]["content"])