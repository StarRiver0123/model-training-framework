from src.models.models.base_component import gen_seq_only_mask, gen_full_false_mask
import torch, numpy

def generate_square_subsequent_mask1(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_square_subsequent_mask2(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


a = torch.from_numpy(numpy.arange(25).reshape(5,5))
print(gen_full_false_mask(a,a))
print(gen_seq_only_mask(a,a))

print(generate_square_subsequent_mask1(5))

print(generate_square_subsequent_mask2(5))