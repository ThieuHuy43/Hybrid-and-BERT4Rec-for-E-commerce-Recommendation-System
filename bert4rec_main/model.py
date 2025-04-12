import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, num_heads=4, max_len=100):
        super(BERT4Rec, self).__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_len,
            type_vocab_size=1,
            pad_token_id=0
        )

        self.bert = BertModel(config)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # attention mask: 1 nếu là token thật, 0 nếu là padding (0)
        attention_mask = (input_ids != 0).long()

        # BERT đầu ra có: last_hidden_state và pooler_output
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Dự đoán item tiếp theo dựa vào token cuối cùng trong chuỗi
        last_token_output = output.last_hidden_state[:, -1, :]  # shape: [batch_size, hidden_size]
        logits = self.fc(last_token_output)  # shape: [batch_size, vocab_size]
        return logits
