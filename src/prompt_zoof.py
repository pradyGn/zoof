import json

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from utils import config_dataclass, encode_input, generate_response_and_decode
from zoof_v1_2.model import zoof_v1_2

# Define the model name to use
zoof_model = "Jiraya/zoof-v1.2-394M-chat"

# Download and process config.json from HF
config_path = hf_hub_download(repo_id=zoof_model, filename="config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config = config_dataclass(config)

# Load the model and move it onto GPU
model = zoof_v1_2.from_pretrained(zoof_model, config=config)
model.to("cuda")

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("Jiraya/zoof-tokenizer")

# This is a navie implementation, will update once I implement kv cache
chat_history_ids = []
print("\n\nHi there! You're chatting with Zoof.")
while True:
    user_input = input("What can I help you with next? (Note: You can type 'exit' anytime to stop.)\n\nUser:")
    if user_input.lower() == "exit":
        print("See you soon!👋")
        break
    user_input_ids = encode_input(tokenizer, chat_history_ids, user_input)

    model_output, model_out_ids = generate_response_and_decode(model, tokenizer, user_input_ids)
    chat_history_ids = user_input_ids.tolist() + model_out_ids
    print(f"Zoof: {model_output}")
