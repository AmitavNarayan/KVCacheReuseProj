import torch
def collate (batch: list[dict], pad_token_id: int):
    maxs = {}
    new_batch = {"answer": None, "prefix": None, "suffix": None, "docs": None, "answer": None}
    for example in batch:
        for key in example:
            if key not in maxs: maxs[key] = 0
            if key=="docs":
                maxs[key] = max(maxs[key], max([doc.shape[0] for doc in example[key]]))
            else: 
                maxs[key] = max(maxs[key], example[key].shape[0])
    for example in batch:
        for key in example:
            if key=="docs":
                example[key] = [torch.cat((doc, torch.full((maxs[key] - doc.shape[0],), pad_token_id))) for doc in example[key]]
                example[key] = torch.stack(example[key])
            else:
                example[key] = torch.cat((example[key], torch.zeros(maxs[key] - example[key].shape[0], dtype=torch.int)))
    for key in new_batch:
        new_batch[key] = torch.stack([example[key] for example in batch])
    return new_batch
