import sys
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)
SUFFIX = " Think step by step, then give the final answer after 'Answer:'."
qs = ["What is 47 * 6?",
      "Sam has 40 apples. He buys 25 more and then gives away 30. How many apples does Sam have now?"]
for q in qs:
    inp = tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**inp, max_new_tokens=150, do_sample=False)
    print("\n###", q, "\n" + tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))