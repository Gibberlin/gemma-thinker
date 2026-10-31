import re, random, json
from unsloth import FastLanguageModel

random.seed(42)  # same 200 problems as baseline.py
problems = []
for _ in range(200):
    a, b = random.randint(10, 99), random.randint(10, 99)
    op = random.choice(["+", "-"])
    problems.append((f"What is {a} {op} {b}?", a + b if op == "+" else a - b))

model, tok = FastLanguageModel.from_pretrained(
    "outputs/lora", max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"

SUFFIX = " Think step by step, then give the final answer after 'Answer:'."
BATCH, correct, log = 20, 0, []
for i in range(0, 200, BATCH):
    chunk = problems[i:i + BATCH]
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": q + SUFFIX}],
        add_generation_prompt=True, tokenize=False) for q, _ in chunk]
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=120, do_sample=False)
    for (q, ans), seq in zip(chunk, out):
        text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        nums = re.findall(r"-?\d+", text.split("Answer:")[-1])
        ok = bool(nums) and int(nums[-1]) == ans
        correct += ok
        log.append({"q": q, "truth": ans, "out": text, "ok": ok,
                    "backtracked": "Wait" in text})
    print(f"{i + BATCH}/200 done, running correct: {correct}", flush=True)

bt = [x for x in log if x["backtracked"]]
print(f"\nTuned accuracy: {correct}/200 = {correct/2:.1f}%  (baseline was 70.0%)")
print(f"Backtracked on {len(bt)} problems, {sum(x['ok'] for x in bt)} of those correct")
json.dump(log, open("tuned_log.json", "w"), indent=1)