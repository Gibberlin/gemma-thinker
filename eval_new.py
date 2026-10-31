import re, random, sys
from unsloth import FastLanguageModel

random.seed(321)
cats = {"3-term arithmetic": [], "exact division": [], "4-digit add": []}
for _ in range(40):
    a, b, c = random.randint(10, 99), random.randint(10, 99), random.randint(10, 99)
    cats["3-term arithmetic"].append((f"What is {a} + {b} - {c}?", a + b - c))
    x, y = random.randint(10, 99), random.randint(2, 9)
    cats["exact division"].append((f"What is {x * y} / {y}?", x))
    a, b = random.randint(1000, 9999), random.randint(1000, 9999)
    cats["4-digit add"].append((f"What is {a} + {b}?", a + b))

model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
SUFFIX = " Think step by step, then give the final answer after 'Answer:'."

for name, probs in cats.items():
    correct = 0
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
    print(f"[{sys.argv[1]}] {name}: {correct}/{len(probs)} correct", flush=True)