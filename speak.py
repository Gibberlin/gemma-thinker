import sys
import pyttsx3
from unsloth import FastLanguageModel

def say(text):
    engine = pyttsx3.init()          # fresh engine each time avoids a known Windows hang
    engine.setProperty("rate", 170)
    engine.say(text)
    engine.runAndWait()

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/lora_rft2"
model, tok = FastLanguageModel.from_pretrained(path, max_seq_length=512, load_in_4bit=False)
FastLanguageModel.for_inference(model)

while True:
    q = input("\nYou: ")
    if q.strip().lower() in ("exit", "quit"):
        break
    if not q.strip():
        continue
    inp = tok.apply_chat_template([{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    reply = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print("Model:", reply)
    say(reply)