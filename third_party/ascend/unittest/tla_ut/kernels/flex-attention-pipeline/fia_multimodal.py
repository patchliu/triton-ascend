"""Exact Ascend C FIA composition for the multimodal mask in this benchmark."""


def prepare_call(q, k, v, segments, window, global_prefix):
    import torch
    import torch_npu

    before, image, after = segments
    doc = before + image + after
    if doc <= 0 or image <= 0 or window <= 0 or not 0 <= global_prefix <= doc:
        raise ValueError('Expected a nonempty image, positive window and in-document prefix')
    if q.shape[2] != k.shape[2] or q.shape[2] % doc:
        raise ValueError('Expected equal Q/KV lengths containing whole documents')

    def key_positions(intervals):
        return torch.cat([torch.arange(begin, end) for begin, end in intervals])

    def text_tiles(start, stop):
        for begin in range(start, stop, window):
            end = min(begin + window, stop)
            key_begin = max(0, begin - window)
            intervals = tuple((a, b) for a, b in ((0, min(global_prefix, key_begin)), (key_begin, end)) if a < b)
            keys = key_positions(intervals)
            distance = torch.arange(begin, end)[:, None] - keys[None, :]
            masked = (distance < 0) | ((distance > window) & (keys[None, :] >= global_prefix))
            yield begin, end, intervals, masked[None, None].to(q.device)

    image_end = before + image
    text_before = tuple(text_tiles(0, before))
    text_after = tuple(text_tiles(image_end, doc))
    image_keys = tuple((a, b) for a, b in ((0, min(global_prefix, before)), (before, image_end)) if a < b)
    local_begin = max(global_prefix, before - window)
    local_end = min(image_end, before + window)
    previous_text = None
    if local_begin < before:
        distance = torch.arange(before, local_end)[:, None] - torch.arange(local_begin, before)[None, :]
        previous_text = ((local_begin, before), (distance > window)[None, None].to(q.device))

    def slice_keys(x, base, intervals):
        parts = [x[:, :, base + begin:base + end] for begin, end in intervals]
        return parts[0].contiguous() if len(parts) == 1 else torch.cat(parts, dim=2)

    def attend(base, begin, end, intervals, mask=None, lse=False):
        q_tile = q[:, :, base + begin:base + end].contiguous()
        return torch_npu.npu_fused_infer_attention_score(
            q_tile, slice_keys(k, base, intervals), slice_keys(v, base, intervals),
            num_heads=q.shape[1], num_key_value_heads=k.shape[1], input_layout='BNSD',
            scale=q.shape[-1]**-0.5, atten_mask=mask, softmax_lse_flag=lse,
        )

    def call():
        outputs = []
        for base in range(0, q.shape[2], doc):
            outputs.extend(attend(base, *tile)[0] for tile in text_before)
            image_out, image_lse = attend(base, before, image_end, image_keys, lse=previous_text is not None)
            if previous_text is not None:
                interval, mask = previous_text
                local_out, local_lse = attend(base, before, local_end, (interval,), mask, lse=True)
                count = local_end - before
                dense_lse = image_lse[:, :, :count]
                total = torch.logaddexp(dense_lse, local_lse)
                merged = (image_out[:, :, :count].float() * (dense_lse - total).exp()
                          + local_out.float() * (local_lse - total).exp())
                outputs.extend((merged.to(q.dtype), image_out[:, :, count:]))
            else:
                outputs.append(image_out)
            outputs.extend(attend(base, *tile)[0] for tile in text_after)
        return torch.cat(outputs, dim=2)

    return call
