import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=1024,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

msgs = [{"role": "user", "content": "What is 17 + 25? Think step by step."}]
inputs = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))