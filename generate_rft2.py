import random, re, json
from unsloth import FastLanguageModel

random.seed(2027)
NAMES = ["Maya", "Omar", "Lena", "Raj", "Zoe"]
ITEMS = ["pencils", "marbles", "stickers", "books", "coins"]
TYPES = ["add2", "sub2", "add3", "sub3", "sub3", "mul", "mul", "word", "term3", "add4"]

def make():
    t = random.choice(TYPES)
    if t in ("add2", "sub2", "add3", "sub3"):
        lo, hi = (10, 99) if t.endswith("2") else (100, 999)
        a, b, op = random.randint(lo, hi), random.randint(lo, hi), ("+" if t.startswith("add") else "-")
        return t, f"What is {a} {op} {b}?", a + b if op == "+" else a - b
    if t == "mul":
        a, b = random.randint(10, 99), random.randint(2, 9)
        return t, f"What is {a} * {b}?", a * b
    if t == "term3":
        a, b, c = random.randint(10, 99), random.randint(10, 99), random.randint(10, 99)
        return t, f"What is {a} + {b} - {c}?", a + b - c
    if t == "add4":
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        return t, f"What is {a} + {b}?", a + b
    n, i = random.choice(NAMES), random.choice(ITEMS)
    form = random.choice([1, 2, 3])
    if form == 3:
        a, b = random.randint(2, 9), random.randint(10, 30)
        return "word", f"{n} has {a} bags with {b} {i} in each. How many {i} does {n} have in total?", a * b
    a, b = random.randint(20, 90), random.randint(10, 50)
    c = random.randint(5, a + b - 1)
    if form == 1:
        return "word", f"{n} has {a} {i}. {n} finds {b} more and then loses {c}. How many {i} does {n} have now?", a + b - c
    return "word", f"A box holds {a} {i}. {b} are added and {c} are taken out. How many {i} are in the box?", a + b - c

SUFFIX = " Think step by step, then give the final answer after 'Answer:'."
N_PROBLEMS, K, BATCH = 1200, 4, 16
probs = [make() for _ in range(N_PROBLEMS)]

model, tok = FastLanguageModel.from_pretrained("outputs/lora_rft", max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"

rows, stats = [], {}
for s in range(0, N_PROBLEMS, BATCH):
    chunk = probs[s:s + BATCH]
    texts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}],
             add_generation_prompt=True, tokenize=False) for _, q, _ in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=150, do_sample=True, temperature=0.7,
                         top_p=0.95, num_return_sequences=K)
    cands = [[] for _ in chunk]
    for idx, seq in enumerate(out):
        text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "Answer:" not in text:
            continue
        nums = re.findall(r"-?\d+", text.split("Answer:")[-1])
        if nums and int(nums[-1]) == chunk[idx // K][2]:
            cands[idx // K].append(text)
    for (t, q, a), c in zip(chunk, cands):
        st = stats.setdefault(t, [0, 0]); st[1] += 1
        if c:
            st[0] += 1
            best = min(c, key=len)
            rows.append({"messages": [{"role": "user", "content": q + SUFFIX},
                                      {"role": "assistant", "content": best}]})
    print(f"{min(s + BATCH, N_PROBLEMS)}/{N_PROBLEMS} problems, kept {len(rows)}", flush=True)

with open("train_rft2.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("solved at least once in 4 tries:", {k: f"{v[0]}/{v[1]}" for k, v in stats.items()})