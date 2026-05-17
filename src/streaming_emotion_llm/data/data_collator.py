"""Original-style data collator for stream text, frame features, and labels."""

from functools import partial

import torch
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother


def data_collator(batch: list[tuple], *, tokenizer: PreTrainedTokenizer, **kwargs) -> dict:
    batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_eval_kwargs = zip(*batch)
    tokenized = tokenizer(
        list(batch_text),
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    labels = torch.full_like(tokenized.input_ids, LabelSmoother.ignore_index, dtype=torch.long)

    for item_labels, item_input_ids, offset_mapping, learn_ranges in zip(
        labels,
        tokenized.input_ids,
        tokenized.offset_mapping,
        batch_learn_ranges,
    ):
        for learn_range in learn_ranges:
            start = torch.nonzero(offset_mapping[:, 0] == learn_range.start).flatten()[0].item()
            if offset_mapping[:, 0][-1] >= learn_range.stop:
                stop = torch.nonzero(offset_mapping[:, 0] == learn_range.stop).flatten()[0].item()
            else:
                stop = len(item_input_ids)
            item_labels[start - 1 : stop - 1] = item_input_ids[start:stop]
            item_labels[item_labels >= len(tokenizer) - 1] = tokenizer.eos_token_id

    tokenized["labels"] = labels
    tokenized.pop("offset_mapping")
    tokenized["frames"] = torch.cat(list(batch_frames))
    tokenized["sample_idxs"] = torch.tensor(batch_sample_idx)
    if batch_eval_kwargs[0]:
        tokenized["evaluation_kwargs"] = batch_eval_kwargs[0]
    return tokenized


def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)
