from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/gemma-3-270m-it", max_seq_length=512, load_in_4bit=False)

# LoRA: train small adapter matrices instead of all 270M weights
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

ds = load_dataset("json", data_files="train.jsonl", split="train")
ds = ds.map(lambda b: {"text": [
    tok.apply_chat_template(m, tokenize=False).removeprefix("<bos>")
    for m in b["messages"]]}, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", dataset_num_proc=1,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        num_train_epochs=1, learning_rate=2e-4, warmup_steps=10,
        lr_scheduler_type="linear", optim="adamw_8bit",
        logging_steps=10, seed=3407, report_to="none", output_dir="outputs"))

# only learn from the assistant's reply, not the question
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n")

trainer.train()
model.save_pretrained("outputs/lora")
tok.save_pretrained("outputs/lora")
print("saved to outputs/lora")