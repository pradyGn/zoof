import torch

from config import zoofv1Config


def config_dataclass(config_dict):
    """
    Converts a configuration dictionary into a structured zoofv1Config object.

    Args:
        config_dict (dict): A dictionary containing model configuration parameters.
                            Keys must match the fields defined in zoofv1Config.

    Returns:
        zoofv1Config: An initialized configuration object.
    """
    return zoofv1Config(**config_dict)


def encode_input(tokenizer, user_input: str):
    """
    Preprocesses and tokenizes user input for the model.

    This function formats the input with the '### Instruction' template,
    appends the specific End Of Prompt token (47791), and prepares a CUDA-ready tensor.

    Args:
        tokenizer: The tokenizer object (e.g., Hugging Face AutoTokenizer).
        user_input (str): The raw input string from the user.

    Returns:
        torch.Tensor: A tensor of shape (1, seq_len) containing the input IDs
                      on the 'cuda' device.
    """
    ids = tokenizer.encode(f"### Instruction\n{user_input}")
    ids.append(47791)  # Append custom End-of-Line or separator token
    ids = torch.tensor(ids).view(1, -1)
    ids = ids.to("cuda")
    return ids


def generate_response_and_decode(model, tokenizer, ids):
    """
    Generates a model response and decodes the output tokens.

    This function handles the generation loop with a retry mechanism for robustness.
    It generates up to 1024 new tokens using a temperature of 1.

    Args:
        model: The PyTorch language model for generation.
        tokenizer: The tokenizer used to decode the output IDs back to text.
        ids (torch.Tensor): The input token IDs tensor (shape: 1, seq_len).

    Returns:
        str: The decoded response string.
        None: If generation fails after the maximum number of retries.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            model_out_ids = model.generate(ids, 1024, temperature=1, eos_tok=47790)[0].tolist()[ids.size(1) : -1]
            model_out = tokenizer.decode(model_out_ids)

            ## clean (a bit) the model output
            bad_output_pos = model_out.find("### Instruction")
            model_out = model_out[:bad_output_pos]

            return model_out

        except Exception:
            if attempt == max_retries - 1:
                print("Invalid input, please prompt again!")
                return None
