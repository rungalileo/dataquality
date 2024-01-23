from typing import List
from warnings import warn

import numpy as np
from tqdm.auto import tqdm


def isolate_response_tokens(
    tokenized_formatted_prompt: List[int], response_template: List[int]
) -> List[int]:
    """Identify the final instance of the response_template and use that to isolate just
    the response tokens.

    tokenized_formatted_prompt has = [num_tokens]

    We search for the *final* occurrence of the response_template within the formatted
    prompt through sublist matching. After isolating the final response_template, we
    slice off the remaining tokens, representing the tokenized response.

    Example:
          >> tokenized_formatted_prompt = [[7, 1, 2, 3, 8, 5, 9, 1, 2, 3, 9, 10, 6]]
          >> response_template = [1, 2, 3]
          >> extract_tokenized_responses(tokenized_formatted_prompt, response_template)
            [[9, 10, 6]]

    If a sample does not contain the response_template we represent the
    tokenized_response for that sample as [] - i.e. the <Empty String>.
    """
    # Reverse search over matches of the first token in the response template
    matched_indices = np.where(
        np.array(tokenized_formatted_prompt) == response_template[0]
    )[0]
    response_token_ids_start_idx = None
    for match_idx in reversed(matched_indices):
        # match_idx = matched_indices[-(i + 1)]
        # Check for exact match of the response template token ids.
        # Once found break to avoid finding further matches + short-circuit
        # search.
        if (
            tokenized_formatted_prompt[match_idx : match_idx + len(response_template)]
            == response_template
        ):
            response_token_ids_start_idx = match_idx
            break

    tokenized_response = []
    # Warn that the response template was not found!
    if response_token_ids_start_idx is None:
        warn(
            f"Could not find response key `{response_template}` in the "
            f"following instance: `{tokenized_formatted_prompt}`. "
            f"This instance will have an <Empty> Target Output. "
            f"Note, if this happens often, consider increasing `max_seq_length`."
        )
    else:
        response_token_ids_end_idx = response_token_ids_start_idx + len(
            response_template
        )
        tokenized_response = tokenized_formatted_prompt[response_token_ids_end_idx:]

    return tokenized_response


def extract_tokenized_responses(
    tokenized_formatted_prompts: List[List[int]], response_template: List[int]
) -> List[List[int]]:
    """Extracts the tokenized responses from the formatted prompts

    For each sample, we isolate the response (from the input) and return it.
    """
    tokenized_responses: List[List[int]] = []
    for t_prompt in tqdm(
        tokenized_formatted_prompts,
        leave=False,
        desc="Identifying `response_template` to isolate response token.",
    ):
        tokenized_response = isolate_response_tokens(t_prompt, response_template)
        tokenized_responses.append(tokenized_response)

    return tokenized_responses
