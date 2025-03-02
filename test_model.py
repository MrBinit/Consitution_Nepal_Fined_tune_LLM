from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch


base_model = "/home/binit/fine_tune_LLama/Llama-3.2-3B_fined_tuned/checkpoint-3500"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [{"role": "user", "content": "नेपाल सरकारले आफ्ना नागरिकलाई दिएको मौलिक हक के हो ?"}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = pipe(prompt, max_new_tokens=1024, do_sample=True)

print(outputs[0]["generated_text"])