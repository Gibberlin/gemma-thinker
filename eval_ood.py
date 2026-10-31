import re, random, sys
from unsloth import FastLanguageModel

random.seed(123)
cats = {"3-digit add/sub": [], "multiplication": [], "word problem": []}
for _ in range(40):
    a, b = random.randint(100, 999), random.randint(100, 999)
    op = random.choice(["+", "-"])
    cats["3-digit add/sub"].append((f"What is {a} {op} {b}?", a + b if op == "+" else a - b))
    a, b = random.randint(10, 99), random.randint(2, 9)
    cats["multiplication"].append((f"What is {a} * {b}?", a * b))
    a, b = random.randint(20, 90), random.randint(10, 50)
    c = random.randint(5, a + b - 1)
    cats["word problem"].append((f"Sam has {a} apples. He buys {b} more and then gives away {c}. How many apples does Sam have now?", a + b - c))

model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
SUFFIX = " Think step by step, then give the final answer after 'Answer:'."

for name, probs in cats.items():
    correct, bt = 0, 0
    for i in range(0, len(probs), 20):
        chunk = probs[i:i + 20]
        texts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}],
                 add_generation_prompt=True, tokenize=False) for q, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=150, do_sample=False)
        for (q, ans), seq in zip(chunk, out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"-?\d+", text.split("Answer:")[-1])
            correct += bool(nums) and int(nums[-1]) == ans
            bt += "Wait" in text
    print(f"[{sys.argv[1]}] {name}: {correct}/{len(probs)} correct, backtracked {bt} times", flush=True)