import logging

import torch
import transformers


_LOGGER = logging.getLogger(__name__)


def _init_weights(module, std=0.02):
    """Huggingface transformer weight initialization, most commonly for bert initialization"""
    if isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class EncoderLanguageBERT(torch.nn.Module):
    def __init__(self, name_weights="bert-base-uncased", num_channels_out=768, num_layers=4, num_heads=12, type_vocab_size=2, max_num_tokens=50):
        super().__init__()

        self.max_num_tokens = max_num_tokens
        self.model_transformers = None
        self.module_projection = None
        self.name_weights = name_weights
        self.num_channels_out = num_channels_out
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokenizer = None
        self.type_vocab_size = type_vocab_size

        self._init()

    def _init(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.name_weights, do_lower_case=True)

        bert_config = transformers.BertConfig(hidden_size=self.num_channels_out, num_hidden_layers=self.num_layers, num_attention_heads=self.num_heads, type_vocab_size=self.type_vocab_size)
        self.model_transformers = transformers.BertModel.from_pretrained(self.name_weights, config=bert_config)

        self.apply(_init_weights)

    def forward(self, inpt):
        tokens_sentence = self.tokenizer(inpt["sentence"], max_length=self.max_num_tokens, add_special_tokens=True, truncation=True, padding="max_length", return_tensors="pt")

        ids_token = tokens_sentence["input_ids"].to(inpt["device"])
        masks_attention_token = tokens_sentence["attention_mask"].bool().to(inpt["device"])

        output_embeddings = self.model_transformers(ids_token, masks_attention_token).last_hidden_state

        output = dict(
            embeddings=output_embeddings,
            mask=masks_attention_token,
        )
        return output
