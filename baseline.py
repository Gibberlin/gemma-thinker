import re, random, json
from unsloth import FastLanguageModel

random.seed(42)
problems = []
for _ in range(200):
    a, b = random.randint(10, 99), random.randint(10, 99)
    op = random.choice(["+", "-"])
    problems.append((f"What is {a} {op} {b}?", a + b if op == "+" else a - b))

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/gemma-3-270m-it", max_seq_length=1024, load_in_4bit=False)
FastLanguageModel.for_inference(model)

correct, log = 0, []
for q, ans in problems:
    msgs = [{"role": "user", "content": q + " Think step by step, then give the final answer after 'Answer:'."}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    nums = re.findall(r"-?\d+", text.split("Answer:")[-1])
    ok = bool(nums) and int(nums[-1]) == ans
    correct += ok
    log.append({"q": q, "truth": ans, "out": text, "ok": ok})

print(f"Baseline accuracy: {correct}/200 = {correct/2:.1f}%")
json.dump(log, open("baseline_log.json", "w"), indent=1)