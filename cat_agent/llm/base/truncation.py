"""Input message truncation to fit within a token budget.

The main entry point is :func:`truncate_input_messages_roughly`.
"""

import copy
from collections import defaultdict
from typing import List

from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, Message
from cat_agent.log import logger
from cat_agent.utils.message_utils import extract_text_from_message
from cat_agent.utils.tokenization_qwen import tokenizer


def truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    """Truncate *messages* so the total token count fits within *max_tokens*.

    The truncation strategy prioritises keeping the system message, the latest
    user turn, and the most recent tool results intact.
    """
    from cat_agent.llm.base.model import ModelServiceError  # deferred

    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise ModelServiceError(
            code='400',
            message='The input messages must contain no more than one system message. '
            ' And the system message, if exists, must be the first message.',
        )
    if not messages:
        return messages

    turns = []
    for m in messages:
        if m.role == SYSTEM:
            continue
        elif m.role == USER:
            turns.append([m])
        else:
            if turns:
                turns[-1].append(m)
            else:
                raise ModelServiceError(
                    code='400',
                    message='The input messages (excluding the system message) must start with a user message.',
                )

    # Build token counts and per-user-turn indices
    available_token = max_tokens
    message_tokens = defaultdict(int)
    last_user_idx = None
    indexed_messages_per_user = defaultdict(list)
    new_messages = []
    for msg_idx, msg in enumerate(messages):
        if msg.role == SYSTEM:
            new_messages.append(msg)
            available_token = max_tokens - _count_tokens(msg)
            continue
        message_tokens[msg_idx] = _count_tokens(msg)
        if msg.role == USER:
            last_user_idx = msg_idx
        indexed_messages_per_user[last_user_idx].append([msg_idx, msg])

    all_tokens = sum(message_tokens.values())
    logger.info(f'ALL tokens: {all_tokens}, Available tokens: {available_token}')
    if all_tokens <= available_token:
        return messages
    if available_token <= 0:
        raise ModelServiceError(
            code='400',
            message=f'The input system has exceed the maximum input context length ({max_tokens} tokens)',
        )

    exceedance = all_tokens - available_token
    for it, (user_msg_idx, indexed_messages) in enumerate(indexed_messages_per_user.items()):
        logger.debug(f'user_msg_idx: {user_msg_idx}, exceedance: {exceedance}')
        if exceedance <= 0:
            new_messages += [x[1] for x in indexed_messages]
        else:
            is_last_turn = (it == len(indexed_messages_per_user) - 1)
            new_turn, exceedance = _truncate_turn(
                indexed_messages1=indexed_messages,
                message_tokens1=message_tokens,
                exceedance=exceedance,
                is_last_turn=is_last_turn,
            )
            if new_turn:
                new_messages += new_turn

    return new_messages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_tokens(msg: Message) -> int:
    if msg.role == ASSISTANT and msg.function_call:
        return tokenizer.count_tokens(f'{msg.function_call}')
    return tokenizer.count_tokens(extract_text_from_message(msg, add_upload_info=True))


def _truncate_message(msg: Message, max_tokens: int, keep_both_sides: bool = False):
    if isinstance(msg.content, str):
        content = tokenizer.truncate(msg.content, max_token=max_tokens, keep_both_sides=keep_both_sides)
    else:
        text = []
        for item in msg.content:
            if not item.text:
                return None
            text.append(item.text)
        text = '\n'.join(text)
        content = tokenizer.truncate(text, max_token=max_tokens, keep_both_sides=keep_both_sides)
    return Message(role=msg.role, content=content)


def _truncate_turn(indexed_messages1: list, message_tokens1: dict, exceedance: int, is_last_turn: bool):
    """Try to reduce *exceedance* by shrinking or removing messages in one turn.

    Four cascading strategies are attempted in order:
      1. Minimise function-call results (except in the last step of the last turn).
      2. Remove entire middle steps.
      3. Truncate function results in the last step.
      4. Truncate user/assistant content.
    """
    all_tokens = sum(message_tokens1[idx] for idx, _ in indexed_messages1)
    logger.debug(f'exceedance start: {exceedance}, all tokens of this turn {all_tokens}')

    if all_tokens <= exceedance:
        return [], (exceedance - all_tokens)

    if len(indexed_messages1) == 1:
        assert is_last_turn
        idx, msg = indexed_messages1[0]
        msg = _truncate_message(msg=msg, max_tokens=message_tokens1[idx] - exceedance, keep_both_sides=True)
        return [msg], 0

    indexed_messages1 = copy.deepcopy(indexed_messages1)
    message_tokens1 = copy.deepcopy(message_tokens1)

    # Split turn into steps (consecutive user / assistant+fn blocks)
    messages_per_step = _split_turn_into_steps(indexed_messages1)
    last_step_idx = messages_per_step[-1][0][0]

    # --- Step 1: minimise function results --------------------------------
    logger.debug(f'exceedance step1 **minimised function result**: {exceedance}')
    for i, (msg_idx, msg) in enumerate(indexed_messages1):
        if exceedance <= 0 or msg.role != FUNCTION or (is_last_turn and msg_idx >= last_step_idx):
            continue
        fn_tok = message_tokens1[msg_idx]
        if fn_tok > exceedance:
            msg = _truncate_message(msg=msg, max_tokens=fn_tok - exceedance, keep_both_sides=True)
            indexed_messages1[i][1] = msg
            message_tokens1[msg_idx] = fn_tok - exceedance
            exceedance = 0
            break
        else:
            msg.content = 'omit'
            message_tokens1[msg_idx] = 0
            exceedance -= fn_tok
    if exceedance <= 0:
        return [x[1] for x in indexed_messages1], 0

    # --- Step 2: remove middle steps --------------------------------------
    logger.debug(f'exceedance step2 **rm middle step**: {exceedance}')
    keep_idx = 0
    for i, step in enumerate(messages_per_step):
        if i == 0 or i == (len(messages_per_step) - 1):
            continue
        step_tokens = sum(message_tokens1[x[0]] for x in step)
        if step_tokens >= exceedance:
            exceedance = 0
            keep_idx = messages_per_step[i + 1][0][0]
            break
        else:
            exceedance -= step_tokens
            keep_idx = messages_per_step[i + 1][0][0]

    if exceedance <= 0:
        res = [x[1] for x in messages_per_step[0]] + [x[1] for x in indexed_messages1 if x[0] >= keep_idx]
        return res, 0

    # --- Step 3: truncate FUNCTION of last step ---------------------------
    logger.debug(f'exceedance step3 **truncate FUNCTION of last step**: {exceedance}')
    messages_to_keep = []
    for msg_idx, msg in messages_per_step[-1]:
        if msg.role != FUNCTION:
            messages_to_keep.append([msg_idx, msg])
            continue
        fn_tok = message_tokens1[msg_idx]
        if fn_tok > exceedance:
            msg = _truncate_message(msg=msg, max_tokens=fn_tok - exceedance, keep_both_sides=True)
            exceedance = 0
        else:
            msg.content = 'omit'
            message_tokens1[msg_idx] = 0
            exceedance -= fn_tok
        messages_to_keep.append([msg_idx, msg])

    messages_to_keep = messages_per_step[0] + messages_to_keep
    if exceedance <= 0:
        return [x[1] for x in messages_to_keep], 0

    # --- Step 4: truncate user/assistant content --------------------------
    logger.debug(f'exceedance step4 **truncate user/assistant content**: {exceedance}')
    for i, (msg_idx, msg) in enumerate(messages_to_keep):
        fn_tok = message_tokens1[msg_idx]
        if fn_tok > exceedance:
            msg = _truncate_message(msg=msg, max_tokens=fn_tok - exceedance, keep_both_sides=True)
            messages_to_keep[i][1] = msg
            exceedance = 0
            break
        else:
            msg.content = 'omit'
            exceedance -= fn_tok

    return [x[1] for x in messages_to_keep], 0


def _split_turn_into_steps(indexed_messages):
    """Group indexed messages by consecutive role (user block, assistant+fn block, ...)."""
    steps = []
    for msg_idx, msg in indexed_messages:
        if msg.role == USER:
            if steps and steps[-1][-1][1].role == USER:
                steps[-1].append([msg_idx, msg])
            else:
                steps.append([[msg_idx, msg]])
        elif msg.role == ASSISTANT:
            if steps and steps[-1][-1][1].role == ASSISTANT:
                steps[-1].append([msg_idx, msg])
            else:
                steps.append([[msg_idx, msg]])
        elif msg.role == FUNCTION:
            steps[-1].append([msg_idx, msg])
    return steps
