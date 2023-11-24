import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaTokenizer, DebertaV2Model

import argparse

# Argument parser setup for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_pt", '-m', type=str, default='')
parser.add_argument("--tokenizer_path", '-t', type=str, default='')
parser.add_argument("--device", type=str, default='cuda')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(36)

test_device = args.device


# Define the PARA model class
class PARAModel(nn.Module):
    def __init__(self,
                 vocab_size=25,
                 hidden_size=512,
                 num_hidden_layers=12,
                 num_attention_heads=8,
                 pad_token_id=22
                 ):
        super().__init__()
        self.embed_dim = hidden_size
        model_cofig1 = DebertaV2Config()
        model_cofig1.hidden_size = hidden_size
        model_cofig1.pad_token_id = pad_token_id
        model_cofig1.pooler_hidden_size = hidden_size
        model_cofig1.intermediate_size = hidden_size * 4
        model_cofig1.pos_att_type = ["p2c", "c2p"]
        model_cofig1.relative_attention = True
        model_cofig1.num_hidden_layers = num_hidden_layers
        model_cofig1.num_attention_heads = num_attention_heads
        model_cofig1.vocab_size = vocab_size
        model_cofig1.position_biased_input = False
        self.encoder = DebertaV2Model(config=model_cofig1)
        self.cls = nn.Linear(hidden_size, vocab_size, bias=False)
        cls_bias = nn.Parameter(torch.zeros(vocab_size))
        self.cls.bias = cls_bias
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def predict(self, x, attention_mask):
        outputs = self.encoder(x,
                               attention_mask=attention_mask,
                               return_dict=True,
                               )
        last_hidden_state = outputs.last_hidden_state
        logits = self.cls(last_hidden_state)
        return last_hidden_state, logits


# Define the testing class for PARA model
class PARATest(nn.Module):
    def __init__(self, device=test_device):
        super().__init__()
        self.model = PARAModel(hidden_size=512, num_attention_heads=8, num_hidden_layers=12, ).to(device)
        ckpt = torch.load(args.model_pt, map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        tokenizer_path = args.tokenizer_path
        self.tokenizer = DebertaTokenizer.from_pretrained(tokenizer_path, )
        self.device = device

    def token_encode(self, sequences):
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"
        sequences = ["".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(sequences,
                                       padding='max_length',
                                       max_length=150,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors="pt",
                                       )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)
        return tokens, attention_mask

    def forward(self, sequences):
        tokens, attention_mask = self.token_encode(sequences)
        with torch.no_grad():
            last_embeds, outputs = self.model.predict(
                tokens,
                attention_mask,
            )
        prob = outputs[:, 1:-1].softmax(-1)
        return last_embeds, outputs, prob


if __name__ == '__main__':
    one_test = PARATest()
    label_data_lst = [
        'EVQLVESGGGLVQPGRSLKLSCAASGFTFSNYYMAWVRQAPKKGLEWVATISTSGSRTYYPDSVKGRFTISRDNAKSSLYLQMNSLKSEDTATYYCATSLITNYWYFDFWGPGTMVTVSS',
        'EVQLVESGGGLVQPGRSLKLSCAASGFTFSDYNMAWVRQAPKKGLEWVATISYDGSSTYYRDSVKGRFTISRDNAKSTLYLQMDSLRSEDTATYYCARHRWFNYGSPFMDAWGQGASVTVSS',
        'EVQLVETGGGLVQPGKSLKLTCATSGFTFSTAWMHWVRQSPEKRLEWIARIKDKSNNYATDYVESVKGRFTISRDDSKSSVYLQMNSLKEEDTATYYCKAPLFDYWGQGVMVTVSS',
        'EVQLHQSGAELVKPGVSVKISCKASGYSFTSYNMHWVKQRPGQAVEWIGVINPESGGTDYNGKFRGKVTLTVDKSSSTAFMQLGSLTPEDTAVYYCARQRVIRGRAHWFAYWGQGTLVTVSS',
        'VQLVESGGGLVQPGKSLKLSCSASGFTFSSYGMHWIRQAPGKGLDWVAYISSSSGTVYADAVKGRFTISRDNAKNTLYLQLNSLKSEDTAIYYCARENYGGYSPYFDYWGQGVMVTVSS', ]

    for i in label_data_lst:
        outs = one_test.forward(i)
