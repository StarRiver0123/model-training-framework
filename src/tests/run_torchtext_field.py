from torchtext.legacy.data import Field

field = Field(sequential=True, use_vocab=True, tokenize=None, batch_first=True,
      fix_length=None, init_token='<sos>', eos_token='<eos>',
      pad_token='<pad>', unk_token='<unk>')
print('ok')