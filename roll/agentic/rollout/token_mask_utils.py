from typing import List, Dict

from transformers import PreTrainedTokenizer


def messages_to_tokens_and_masks(messages: List[Dict], tokenizer: PreTrainedTokenizer, add_generation_prompt=False):
    """
    Convert a list of chat messages into token IDs and response masks.

    Args:
        messages: List of dictionaries representing chat messages, each with 'role' and 'content' keys.
        tokenizer: PreTrainedTokenizer instance to tokenize the messages.
        add_generation_prompt: Whether to add a generation prompt for the last message (default: False).

    Returns:
        Tuple[List[List[int]], List[List[int]]]: A tuple containing:
            - List of token ID lists for each message.
            - List of response mask lists (1 for tokens to be predicted, 0 otherwise).

    case:
        messages = [
            {"role": "system",
             "content": "You're a helpful assistant. "},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
            {"role": "assistant", "content": "Landon"}
        ]
        <|im_start|>system
        You're a helpful assistant. <|im_end|>
        <|im_start|>user
        What is the capital of France?<|im_end|>
        <|im_start|>assistant
        Paris<|im_end|>
        <|im_start|>assistant
        Landon<|im_end|>

        [[151644, 8948, 198, 2610, 2299, 264, 10950, 17847, 13, 220, 151645, 198], [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198], [151644, 77091, 198, 59604, 151645, 198], [151644, 77091, 198, 43, 11037, 151645]]
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
        [('<|im_start|>', 0, 151644), ('system', 0, 8948), ('\n', 0, 198), ('You', 0, 2610), ("'re", 0, 2299), (' a', 0, 264), (' helpful', 0, 10950), (' assistant', 0, 17847), ('.', 0, 13), (' ', 0, 220), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('user', 0, 872), ('\n', 0, 198), ('What', 0, 3838), (' is', 0, 374), (' the', 0, 279), (' capital', 0, 6722), (' of', 0, 315), (' France', 0, 9625), ('?', 0, 30), ('<|im_end|>', 0, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('Paris', 1, 59604), ('<|im_end|>', 1, 151645), ('\n', 0, 198), ('<|im_start|>', 0, 151644), ('assistant', 0, 77091), ('\n', 0, 198), ('L', 1, 43), ('andon', 1, 11037), ('<|im_end|>', 1, 151645)]

    """
    token_ids_list = []
    response_masks_list = []
    system_mock = {"role": "system", "content": ""}
    system_token_ids_mock = tokenizer.apply_chat_template([system_mock], tokenize=True)
    for i, message in enumerate(messages):
        if message["role"].lower() == "system":
            token_ids = tokenizer.apply_chat_template([message], tokenize=True)
            token_ids_list.append(token_ids)
            response_masks_list.append([0] * len(token_ids))
        if message["role"].lower() in ["user", "assistant"]:
            token_ids = tokenizer.apply_chat_template([system_mock, message], tokenize=True,
                                                      add_generation_prompt=add_generation_prompt and i == len(messages) - 1)
            token_ids = token_ids[len(system_token_ids_mock):]
            if message["role"].lower() == "user":
                token_ids_list.append(token_ids)
                response_masks_list.append([0] * len(token_ids))
            elif message["role"].lower() == "assistant":
                assistant_token_ids = []
                response_masks = []
                token_id_without_format = tokenizer.encode(message["content"])
                first_assistant_idx = token_ids.index(token_id_without_format[0])
                assistant_token_ids.extend(token_ids[: first_assistant_idx])
                response_masks.extend([0] * first_assistant_idx)
                after_eos_token_id = False
                for idx in range(first_assistant_idx, len(token_ids)):
                    if i == len(messages) - 1 and after_eos_token_id and not add_generation_prompt:
                        break
                    assistant_token_ids.append(token_ids[idx])
                    if after_eos_token_id:
                        response_masks.append(0)
                    else:
                        response_masks.append(1)
                    if token_ids[idx] == tokenizer.eos_token_id:
                        after_eos_token_id = True
                token_ids_list.append(assistant_token_ids)
                response_masks_list.append(response_masks)

    return token_ids_list, response_masks_list


def token_ids_to_assistant_mask(messages: List[Dict], input_ids_list: List[List], tokenizer: PreTrainedTokenizer):
    """
    Reconstruct per-message assistant masks from already-tokenized inputs.

    Args:
        messages: Original chat messages; only the `"assistant"` ones are inspected.
        input_ids_list: Already-tokenized version of `messages`, aligned index-by-index.
        tokenizer: Hugging Face tokenizer whose `.eos_token_id` marks the end of an assistant turn.

    Returns:
        List[List[int]] where each inner list contains 1 for tokens belonging to the assistant’s
        actual response (excluding format / control tokens) and 0 elsewhere.
    """
    token_ids_list = []
    assistant_mask_list = []
    for i, (message, token_ids) in enumerate(zip(messages, input_ids_list)):
        if message["role"].lower() in ["system", "user"]:
            token_ids_list.append(token_ids)
            assistant_mask_list.append([0] * len(token_ids))
        if message["role"].lower() in ["assistant"]:
            assistant_token_ids = []
            assistant_mask = []
            token_id_without_format = tokenizer.encode(message["content"])
            first_assistant_idx = token_ids.index(token_id_without_format[0])
            assistant_token_ids.extend(token_ids[: first_assistant_idx])
            assistant_mask.extend([0] * first_assistant_idx)
            after_eos_token_id = False
            for idx in range(first_assistant_idx, len(token_ids)):
                assistant_token_ids.append(token_ids[idx])
                if after_eos_token_id:
                    assistant_mask.append(0)
                else:
                    assistant_mask.append(1)
                if token_ids[idx] == tokenizer.eos_token_id:
                    after_eos_token_id = True
            token_ids_list.append(assistant_token_ids)
            assistant_mask_list.append(assistant_mask)

    return assistant_mask_list


def split_by_token(input_ids: list, token: int) -> list[list]:
    """
    Split the input_ids list by the given token and return a list of lists.
    Each sub-list starts with that token.

    Args:
        input_ids: Sequence of input IDs to be split (list[int]).
        token: Token ID used for splitting and marking the start of each sub-list.

    Returns:
        A list containing multiple sub-lists (List[List[int]]).

    Example:
      >>> split_by_token(
      ...     input_ids=[151644, 8948, 198, 151644, 872, 198],
      ...     token=151644
      ... )
      [[151644, 8948, 198], [151644, 872, 198]]
    """

    if not input_ids:
        return []

    result = []
    current_segment = []

    for item in input_ids:
        if item == token:
            if current_segment:
                result.append(current_segment)
            current_segment = [item]
        else:
            current_segment.append(item)

    if current_segment:
        result.append(current_segment)

    return result
