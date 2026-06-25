import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from . import constants

class cattleWordEmbedding(nn.Module):
    def __init__(self,
        vocab_size,
        hidden_dim,
        vocab_dim = 768,
        padding_idx=0,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        vocab_freeze=True,
        use_bert=True,
        ) -> None:
        super().__init__()

        # Load pretrained BERT embeddings for column headers
        if use_bert:
            print("using bert")
            word2vec_weight = torch.load('./bert_emb.pt')
            self.word_embeddings_header = nn.Embedding.from_pretrained(word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx)
        
        # Learnable embeddings for feature values
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings_value.weight)

        # LayerNorm for Header Embeddings (with pretrained weights)
        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        if use_bert:
            weight_emb = torch.load('./bert_layernorm_weight.pt')
            bias_emb = torch.load('./bert_layernorm_bias.pt')
            self.norm_header.weight.data.copy_(weight_emb)
            self.norm_header.bias.data.copy_(bias_emb)
            if vocab_freeze:
                print("freeze")
                self.freeze(self.norm_header)

        # LayerNorm for Value Embeddings
        self.norm_value = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)

        # Dropout for Regularization
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input_ids, emb_type='value') -> torch.Tensor:
        """
        emb_type: 'header' for column headers, 'value' for feature values
        """
        if emb_type == 'header':
            embeddings = self.word_embeddings_header(input_ids)
            embeddings = self.norm_header(embeddings)
        elif emb_type == 'value':
            embeddings = self.word_embeddings_value(input_ids)
            embeddings = self.norm_value(embeddings)
        else:
            raise RuntimeError(f'Unknown embedding type: {emb_type}')

        return self.dropout(embeddings)
    
class cattleNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        '''args:
        num_col_emb: numerical column embedding, (# numerical columns, emb_dim)
        x_num_ts: numerical features, (bs, emb_dim)
        num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        '''
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb


class cattleFeatureExtractor:
    r'''
    Process input dataframe to input indices towards cattle encoder,
    usually used to build dataloader for paralleling loading.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        '''args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        disable_tokenizer_parallel: true if use extractor for collator function in torch.DataLoader
        ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
            if set `true`, the duplicate cols will be deleted, else throws errors.
        '''
        if os.path.exists('./cattle/tokenizer'):
            print('using vocab')
            self.tokenizer = BertTokenizerFast.from_pretrained('./cattle/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./cattle/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

    def __call__(self, x, shuffle=False) -> Dict:
        '''
        Parameters
        ----------
        x: pd.DataFrame 
            with column names and features.

        shuffle: bool
            if shuffle column order during the training.

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
                'x_cat_input_ids': tensor contains categorical column + feature ids,
                'x_bin_input_ids': tesnor contains binary column + feature ids,
            }
        '''
        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
            'x_bin_input_ids':None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        # TODO:
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill Nan with zero
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention


        if len(cat_cols) > 0:
            # Convert each categorical column to string and fill missing values
            x_cat = x[cat_cols].astype(str).fillna('')
            # Create a list (per row) where each element is the string from one column.
            x_cat_str = x_cat.values.tolist()  # Each element is a list of column strings for that row
        
            encoded_inputs['x_cat_input_ids'] = []
            encoded_inputs['cat_att_mask'] = []
            max_y = 0
            cat_cnt = len(cat_cols)
            # Determine maximum token length per column (here using 2048/cat_cnt)
            max_token_len = max(1, int(2048 / cat_cnt))
            
            # Process each row independently
            for sample in x_cat_str:
                # Tokenize each column's string separately; results is a dict of tensors.
                tokens = self.tokenizer(sample, padding=True, truncation=True,
                                        add_special_tokens=False, return_tensors='pt')
                # Truncate each column's tokens to max_token_len
                tokens['input_ids'] = tokens['input_ids'][:, :max_token_len]
                tokens['attention_mask'] = tokens['attention_mask'][:, :max_token_len]
                encoded_inputs['x_cat_input_ids'].append(tokens['input_ids'])
                encoded_inputs['cat_att_mask'].append(tokens['attention_mask'])
                max_y = max(max_y, tokens['input_ids'].shape[1])
            
            # Re-pad each row's output so all have the same sequence length across columns
            for i in range(len(encoded_inputs['x_cat_input_ids'])):
                tmp = torch.full((cat_cnt, max_y), self.pad_token_id, dtype=torch.int)
                tmp[:, :encoded_inputs['x_cat_input_ids'][i].shape[1]] = encoded_inputs['x_cat_input_ids'][i]
                encoded_inputs['x_cat_input_ids'][i] = tmp
        
                tmp = torch.zeros((cat_cnt, max_y), dtype=torch.int)
                tmp[:, :encoded_inputs['cat_att_mask'][i].shape[1]] = encoded_inputs['cat_att_mask'][i]
                encoded_inputs['cat_att_mask'][i] = tmp
        
            encoded_inputs['x_cat_input_ids'] = torch.stack(encoded_inputs['x_cat_input_ids'], dim=0)
            encoded_inputs['cat_att_mask'] = torch.stack(encoded_inputs['cat_att_mask'], dim=0)
        
            # Tokenize column headers separately (unchanged)
            col_cat_ts = self.tokenizer(cat_cols, padding=True, truncation=True,
                                        add_special_tokens=False, return_tensors='pt')
            encoded_inputs['col_cat_input_ids'] = col_cat_ts['input_ids']
            encoded_inputs['col_cat_att_mask'] = col_cat_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        '''save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)


    def update(self, cat=None, num=None, bin=None):
        '''update cat/num/bin column maps.
        '''
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(cat))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(num))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(bin))

class cattleMaskToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mask_emb = nn.Parameter(torch.Tensor(hidden_dim))
        nn_init.uniform_(self.mask_emb, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, embeddings: torch.Tensor, mask_indices: torch.Tensor, header_emb: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, T, D) - feature embeddings.
        mask_indices: (B, T) - 1 = masked, 0 = not masked.
        header_emb: (T, D) - column header embeddings.
        
        Returns:
            Modified embeddings where masked positions are replaced by (mask_emb + header_emb).
        """
        # Zero out positions that are masked.
        embeddings[mask_indices.bool()] = 0
        
        bs, fs = embeddings.shape[0], embeddings.shape[1]
        # Expand mask_emb to shape (B, T, D) and add header embeddings expanded to (B, T, D)
        all_mask_token = self.mask_emb.unsqueeze(0).unsqueeze(0).expand(bs, fs, -1) + \
                         header_emb.unsqueeze(0).expand(bs, -1, -1)
        # Add the mask tokens (only at masked positions, since mask_indices.unsqueeze(-1) acts as a selector)
        embeddings = embeddings + all_mask_token * mask_indices.unsqueeze(-1)
        return embeddings


class cattleFeatureProcessor(nn.Module):
    r'''
    Minimal modification to store extra info for masked pretraining.
    '''
    def __init__(self,
                 vocab_size=None,
                 hidden_dim=128,
                 hidden_dropout_prob=0,
                 pad_token_id=0,
                 device='cuda:0'):
        super().__init__()
        self.word_embedding = cattleWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
        )
        self.num_embedding = cattleNumEmbedding(768)
        self.align_layer = nn.Linear(768, hidden_dim, bias=False)
        self.device = device
        # Optionally, if you use pooling policies, you might store it here.
        self.pool_policy = 'avg'

    def _avg_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is None:
            return embs.mean(-2)
        else:
            embs[att_mask == 0] = 0
            embs = embs.sum(-2) / (att_mask.sum(-1, keepdim=True).to(embs.device) + eps)
            return embs

    def _max_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is not None:
            embs[att_mask == 0] = -1e12
        embs = torch.max(embs, dim=-2)[0]
        return embs

    # Optionally, if self-attention pooling is used:
    def _sa_block(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        # Assumes self.self_attn is defined in the model (if needed)
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x[:, 0, :]

    def forward(self,
                x_num=None,
                num_col_input_ids=None,
                num_att_mask=None,
                x_cat_input_ids=None,
                cat_att_mask=None,
                col_cat_input_ids=None,
                col_cat_att_mask=None,
                x_bin_input_ids=None,
                bin_att_mask=None,
                **kwargs):
        """
        Returns
        -------
        A tuple with:
         - A dictionary {'embedding': (bs, total_tokens, hidden_dim),
                         'attention_mask': (bs, total_tokens)}
         - A dictionary with auxiliary info (e.g. col_emb, num_count, etc.)
        """
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None
        other_info = {
            'col_emb': None,         # Combined column embeddings from numerical and categorical features
            'num_count': 0,          # Count of numerical features
            'x_num': x_num,          # (bs, num_fs)
            'cat_bert_emb': None      # (bs, cat_fs, hidden_dim)
        }
    
        if other_info['x_num'] is not None:
            other_info['x_num'] = other_info['x_num'].to(self.device)

        if self.pool_policy == 'avg':
            if x_num is not None and num_col_input_ids is not None:
                # Process numerical features:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header')
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                # Process categorical feature values:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, cat_att_mask)
                # Process categorical column headers:
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                # Expand header embeddings to match batch size:
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0], -1, -1))
                
                # Combine the column header and the value embeddings:
                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)
    
                # Also compute a BERT-based reference for categorical features:
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, cat_att_mask)
    
                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
    
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
    
        # Process binary features if available (remains unchanged)
        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0:
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0], 1, dtype=torch.int)
            bin_emb = self.word_embedding(x_bin_input_ids.to(self.device), emb_type='value')
            bin_emb = self.align_layer(bin_emb)
            bin_mask = torch.ones(bin_emb.shape[:2], device=self.device)
            # Append binary embeddings into the overall list:
            bin_feat_embedding = bin_emb  # (bs, ? , hidden_dim)
    
        # Concatenate numerical, categorical, and binary features:
        emb_list = []
        att_mask_list = []
        col_emb = []
        if num_feat_embedding is not None:
            col_emb.append(num_col_emb)
            other_info['num_count'] = num_col_emb.shape[0]
            emb_list.append(num_feat_embedding)
            att_mask_list.append(torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device))
        if cat_feat_embedding is not None:
            col_emb.append(cat_col_emb)
            emb_list.append(cat_feat_embedding)
            att_mask_list.append(torch.ones(cat_feat_embedding.shape[0], cat_feat_embedding.shape[1]).to(self.device))
        if bin_feat_embedding is not None:
            emb_list.append(bin_feat_embedding)
            att_mask_list.append(torch.ones(bin_feat_embedding.shape[:2], device=self.device))
    
        if len(emb_list) == 0:
            raise Exception('No feature found among numerical, categorical, or binary.')
    
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        other_info['col_emb'] = torch.cat(col_emb, 0).float() if col_emb else None
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}, other_info
 
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

class cattleTransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None) -> Tensor:
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x



class cattleInputEncoder(nn.Module):
    '''
    Build a feature encoder that maps inputs tabular samples to embeddings.
    
    Parameters:
    -----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds 
        multiple workers and the tokenizer builds multiple workers at the same time.

    hidden_dim: int
        the dimension of hidden embeddings.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    '''
    def __init__(self,
        feature_extractor,
        feature_processor,
        device='cuda:0',
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        Encode input tabular samples into embeddings.

        Parameters
        ----------
        x: pd.DataFrame
            with column names and features.        
        '''
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

class cattleEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
            cattleTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )
        if num_layer > 1:
            encoder_layer = cattleTransformerLayer(d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,
                )
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        '''args:
        embedding: bs, num_token, hidden_dim
        '''
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

class cattleLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        #p=None,
        hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        
  
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        #print(x)
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits

class cattleProjectionHead(nn.Module):
    def __init__(self,
        hidden_dim=128,
        projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        x=x.to('cuda:0')
        h = self.dense(x)
        return h

import torch
import torch.nn as nn
import torch.nn.functional as F


class cattleCLSToken(nn.Module):
    '''add a learnable cls token embedding at the end of each sequence.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)
        #return self.weight.view(1,1,1, self.hidden_dim).expand(*leading_dimensions)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        #'''
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs
        #'''

class cattleModel(nn.Module):
    '''The base cattle model for downstream tasks like contrastive learning, binary classification, etc.
    All models subclass this basemodel and usually rewrite the ``forward`` function. Refer to the source code of
    :class:`cattle.modeling_cattle.cattleClassifier` or :class:`cattle.modeling_cattle.cattleForCL` for the implementation details.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: cattleFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A cattleModel model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=5,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None:
            feature_extractor = cattleFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = cattleFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )
        
        self.input_encoder = cattleInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
            )

        self.encoder = cattleEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,#2
            num_attention_head=8,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            )

        self.cls_token = cattleCLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)



    def forward(self, x, y=None):
        '''Extract the embeddings based on input tables.

        Parameters
        ----------
        x: pd.DataFrame
            a batch of samples stored in pd.DataFrame.

        y: pd.Series
            the corresponding labels for each sample in ``x``. ignored for the basemodel.

        Returns
        -------
        final_cls_embedding: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        '''
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        # go through transformers, get final cls embedding
        encoder_output = self.encoder(**embeded)

        # get cls token
        final_cls_embedding = encoder_output[:,0,:]
        return encoder_output #final_cls_embedding

    def load_pretrained_weights(self, ckpt_dir):
        '''Load the model state_dict and feature_extractor configuration
        from the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to load.

        Returns
        -------
        None

        '''
        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns
        

    def save(self, ckpt_dir):
        '''Save the model state_dict and feature_extractor configuration
        to the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to save.

        Returns
        -------
        None

        '''
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        '''Update the configuration of feature extractor's column map for cat, num, and bin cols.
        Or update the number of classes for the output classifier layer.

        Parameters
        ----------
        config: dict
            a dict of configurations: keys cat:list, num:list, bin:list are to specify the new column names;
            key num_class:int is to specify the number of classes for finetuning on a new dataset.

        Returns
        -------
        None

        '''

        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)
        
        if 'p' in config:
            p = config['p']
            self._adapt_to_new_p(config['num_class'], p)

        return None

    def _get_all_attn_layers(self):
        """Recursively collect all self_attn layers from the encoder."""
        attn_layers = []
    
        def _recurse(module):
            if hasattr(module, "self_attn"):
                attn_layers.append(module.self_attn)
            for child in module.children():
                _recurse(child)
    
        for block in self.encoder.transformer_encoder:
            _recurse(block)
    
        return attn_layers
    
    
    def _reset_model_weights(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    
    
    @staticmethod
    def _make_freeze_kv_hook(d_model):
        """Return a standalone hook that zeros K/V gradients without capturing `self`."""
        def hook(grad):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[d_model:] = 0
            return grad
        return hook
    
    
    @staticmethod
    def _make_freeze_kv_bias_hook(d_model):
        """Return a standalone hook that zeros K/V bias gradients."""
        def hook(grad):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[d_model:] = 0
            return grad
        return hook
    
    
    def apply_ca(self, mapping, reset_model=True):
        """
        mapping: dict
            mapping[target_attn_idx] = source_attn_idx
    
        Examples:
            {0: 0}         -> source 0 to target 0
            {0: 0, 1: 1}   -> source 0->target 0, source 1->target 1
            {0: 0, 1: 0}   -> source 0 to target 0 and 1
        """
    
        source_attn_layers = self._get_all_attn_layers()
    
        pretrained_k_weights = []
        pretrained_v_weights = []
        pretrained_k_biases = []
        pretrained_v_biases = []
        source_has_bias = []
    
        for attn in source_attn_layers:
            qkv_weight = attn.in_proj_weight
            d_model = attn.embed_dim
    
            k_weight = qkv_weight[d_model:2 * d_model, :].detach().clone()
            v_weight = qkv_weight[2 * d_model:3 * d_model, :].detach().clone()
            pretrained_k_weights.append(k_weight)
            pretrained_v_weights.append(v_weight)
    
            if attn.in_proj_bias is not None:
                qkv_bias = attn.in_proj_bias
                k_bias = qkv_bias[d_model:2 * d_model].detach().clone()
                v_bias = qkv_bias[2 * d_model:3 * d_model].detach().clone()
                pretrained_k_biases.append(k_bias)
                pretrained_v_biases.append(v_bias)
                source_has_bias.append(True)
            else:
                pretrained_k_biases.append(None)
                pretrained_v_biases.append(None)
                source_has_bias.append(False)
    
        print("Extracted source KV from attention layers:",
              list(range(len(pretrained_k_weights))))
    

        if reset_model:
            self._reset_model_weights()
            print("Model weights reset after cloning source KV.")
    
        target_attn_layers = self._get_all_attn_layers()
    
        if len(target_attn_layers) != len(pretrained_k_weights):
            print(
                f"Warning: number of target attention layers ({len(target_attn_layers)}) "
                f"differs from extracted source layers ({len(pretrained_k_weights)})."
            )
    
        # Store hook handles so they can be removed later if needed
        if not hasattr(self, "_ca_hook_handles"):
            self._ca_hook_handles = []
    
        with torch.no_grad():
            print("CA mapping (target -> source):", mapping)
    
            for tgt_idx, src_idx in mapping.items():
    
                if tgt_idx >= len(target_attn_layers):
                    print(f"Warning: Target attention index {tgt_idx} exceeds available target layers.")
                    continue
    
                if src_idx >= len(pretrained_k_weights):
                    print(f"Warning: Source attention index {src_idx} exceeds available source KV weights.")
                    continue
    
                tgt_attn = target_attn_layers[tgt_idx]
                qkv_weight = tgt_attn.in_proj_weight
                d_model = tgt_attn.embed_dim
    
                # --- Weights ---
                q_weight = qkv_weight[:d_model, :].detach().clone()
                k_weight = pretrained_k_weights[src_idx]
                v_weight = pretrained_v_weights[src_idx]
    
                if q_weight.shape != k_weight.shape or q_weight.shape != v_weight.shape:
                    print(
                        f"Warning: Weight shape mismatch for target {tgt_idx} and source {src_idx}. "
                        f"Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}"
                    )
                    continue
    
                new_qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                tgt_attn.in_proj_weight.data.copy_(new_qkv_weight)
    
                # --- Biases ---
                if source_has_bias[src_idx] and tgt_attn.in_proj_bias is not None:
                    qkv_bias = tgt_attn.in_proj_bias
                    q_bias = qkv_bias[:d_model].detach().clone()
                    k_bias = pretrained_k_biases[src_idx]
                    v_bias = pretrained_v_biases[src_idx]
    
                    if q_bias.shape != k_bias.shape or q_bias.shape != v_bias.shape:
                        print(
                            f"Warning: Bias shape mismatch for target {tgt_idx} and source {src_idx}. "
                            f"Q: {q_bias.shape}, K: {k_bias.shape}, V: {v_bias.shape}"
                        )
                    else:
                        new_qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                        tgt_attn.in_proj_bias.data.copy_(new_qkv_bias)
    
                elif source_has_bias[src_idx] and tgt_attn.in_proj_bias is None:
                    print(
                        f"Warning: Source {src_idx} has bias but target {tgt_idx} does not. "
                        f"Skipping bias transfer."
                    )
                elif not source_has_bias[src_idx] and tgt_attn.in_proj_bias is not None:
                    print(
                        f"Warning: Source {src_idx} has no bias but target {tgt_idx} does. "
                        f"Skipping bias transfer; target bias remains as-is."
                    )
    
                # --- Register freeze hooks (weight) ---
                tgt_attn.in_proj_weight.requires_grad_(True)
    
                if not hasattr(tgt_attn, "_kv_weight_hook_registered"):
                    handle = tgt_attn.in_proj_weight.register_hook(
                        self._make_freeze_kv_hook(d_model)
                    )
                    self._ca_hook_handles.append(handle)
                    tgt_attn._kv_weight_hook_registered = True
    
                # --- Register freeze hooks (bias) ---
                if tgt_attn.in_proj_bias is not None:
                    tgt_attn.in_proj_bias.requires_grad_(True)
    
                    if not hasattr(tgt_attn, "_kv_bias_hook_registered"):
                        handle = tgt_attn.in_proj_bias.register_hook(
                            self._make_freeze_kv_bias_hook(d_model)
                        )
                        self._ca_hook_handles.append(handle)
                        tgt_attn._kv_bias_hook_registered = True
    
                print(f"Transferred source KV from layer {src_idx} -> target layer {tgt_idx}")
    
    
    def remove_ca_hooks(self):
        """Remove all registered KV freeze hooks, allowing K/V to train freely."""
        for handle in getattr(self, "_ca_hook_handles", []):
            handle.remove()
        self._ca_hook_handles = []
    
        for attn in self._get_all_attn_layers():
            attn.__dict__.pop("_kv_weight_hook_registered", None)
            attn.__dict__.pop("_kv_bias_hook_registered", None)
    
        print("All CA freeze hooks removed.")

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = cattleLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

    def _adapt_to_new_p(self, num_class, p):
        if p != self.p:
            self.p = p
            self.clf = cattleLinearClassifier(num_class, hidden_dim=(self.cls_token.hidden_dim*p))
            self.clf.to(self.device)

            
class cattleClassifier(cattleModel):
    '''The classifier model subclass from :class:`cattle.modeling_cattle.cattleModel`.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: cattleFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A cattleClassifier model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=32,
        num_layer=5,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        p=None,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            p=1,
            **kwargs,
        )
        self.p = p
        self.num_class = num_class
        self.clf = cattleLinearClassifier(num_class=num_class, hidden_dim=hidden_dim*p)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)
        
    def partition_and_pad(self, embedding: torch.Tensor, p: int):
        """
        Split the feature dimension (size=n) of `embedding` into up to p chunks,
        each up to ceil(n/p) in length. We then pad the smaller chunk (if any)
        so they have uniform shape. Also returns a matching attention mask
        that has 1 for real tokens and 0 for padded tokens.

        Args:
            embedding: (b, n, d)
            p: number of partitions desired

        Returns:
            parted_emb:  (b, p, chunk_size, d)  # chunk_size = ceil(n/p)
            parted_mask: (b, p, chunk_size)     # 1 for real, 0 for padded
            new_p: how many chunks we actually used (usually = p, but can be less if n < p)
        """
        b, n, d = embedding.shape
        device = embedding.device

        # We compute chunk_size = ceil(n / p). Example: n=13, p=4 => chunk_size=4
        chunk_size = math.ceil(n / p)

        # We'll build a list of partitions (tensors), each up to chunk_size in length
        parted_list = []
        parted_mask_list = []

        start_idx = 0
        for _ in range(p):
            end_idx = min(start_idx + chunk_size, n)
            # If start_idx >= n, we have no more features left
            if start_idx >= n:
                break

            # Slice out [start_idx : end_idx] of the embedding
            chunk = embedding[:, start_idx:end_idx, :]  # shape (b, chunk_len, d)
            chunk_len = chunk.shape[1]

            # Build an attention mask: shape (b, chunk_len) with all ones
            chunk_mask = torch.ones(b, chunk_len, device=device)

            # If the chunk is smaller than chunk_size, pad
            if chunk_len < chunk_size:
                pad_len = chunk_size - chunk_len
                # Pad chunk along dimension=1 (the "feature" dimension)
                # `F.pad` expects (left, right, left, right...) format
                # We'll pad the feature dimension with pad_len zeros on the right
                chunk = F.pad(chunk, (0, 0, 0, pad_len), value=0)  
                # chunk -> (b, chunk_size, d)

                # Also pad the mask with zeros
                chunk_mask = F.pad(chunk_mask, (0, pad_len), value=0)
                # chunk_mask -> (b, chunk_size)

            parted_list.append(chunk)       # shape (b, chunk_size, d)
            parted_mask_list.append(chunk_mask)  # shape (b, chunk_size)

            start_idx = end_idx  # move to next chunk

        # Now stack each chunk along a new dimension => (b, #chunks, chunk_size, d)
        parted_emb = torch.stack(parted_list, dim=1)         # shape (b, new_p, chunk_size, d)
        parted_mask = torch.stack(parted_mask_list, dim=1)   # shape (b, new_p, chunk_size)

        # The actual number of chunks used
        new_p = parted_emb.shape[1]

        return parted_emb, parted_mask, new_p

    
    def forward(self, x, y=None):
        '''Make forward pass given the input feature ``x`` and label ``y`` (optional).

        Parameters
        ----------
        x: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of cattleFeatureExtractor.

        y: pd.Series
            the corresponding labels for each sample in ``x``. if label is given, the model will return
            the classification loss by ``self.loss_fn``.

        Returns
        -------
        logits: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        loss: torch.Tensor or None
            the classification loss.

        '''
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'cattleClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs = self.input_encoder.feature_processor(**inputs)

        embedding_dict, aux_info = outputs 
        outputs = self.cls_token(**embedding_dict)

        encoder_output = self.encoder(**outputs)

        logits = self.clf(encoder_output)

        if y is not None:
            # compute classification loss
            if self.num_class == 2:
                y_ts = torch.tensor(y.values).to(self.device).float()
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss#, qkv_outputs

    
from .trainer_utils import TrainDataset, SupervisedTrainCollator
from torch.utils.data import DataLoader

class MaskedClassifier(cattleClassifier):

    def __init__(self, 
                 cat_cols=None,
                 num_cols=None,
                 bin_cols=None,
                 num_class=2,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0,
                 ffn_dim=256,
                 device='cuda:0',
                 p=1,
                 mlm_probability=0.35,
                 num_rotation=3,  
                 cat_rotation=7,  
                 **kwargs):
        super().__init__(categorical_columns=cat_cols,
                         numerical_columns=num_cols,
                         binary_columns=bin_cols,
                         num_class=num_class,
                         hidden_dim=hidden_dim,
                         num_layer=num_layer,
                         num_attention_head=num_attention_head,
                         hidden_dropout_prob=hidden_dropout_prob,
                         ffn_dim=ffn_dim,
                         device=device,
                         p=p,
                         **kwargs)
        self.mlm_probability = mlm_probability
        self.num_rotation = num_rotation
        self.cat_rotation = cat_rotation
        self.mask_token = cattleMaskToken(hidden_dim)

        # Projection heads for masked reconstruction
        self.num_head = cattleProjectionHead(hidden_dim=hidden_dim*self.p, projection_dim=1)
        self.cat_head = cattleProjectionHead(hidden_dim=hidden_dim*self.p, projection_dim=hidden_dim*self.p)

        self.mse_loss = nn.MSELoss(reduction='mean')

    def mask_features(self, attention_mask, num_count, cat_count):
        """
        Implements adaptive feature masking.
        Ensures numeric & categorical features are masked proportionally.
        """
        if num_count == 0 or cat_count == 0:
            masked_indices = torch.bernoulli(torch.full(attention_mask.shape, self.mlm_probability))
        else:
            num_mp = min(1.0, self.mlm_probability * self.num_rotation * (1 + cat_count / num_count) / 10)
            cat_mp = min(1.0, self.mlm_probability * self.cat_rotation * (1 + num_count / cat_count) / 10)

            if num_mp >= 1.0:
                cat_mp = (self.mlm_probability * (num_count + cat_count) - num_count) / cat_count
            elif cat_mp >= 1.0:
                num_mp = (self.mlm_probability * (num_count + cat_count) - cat_count) / num_count

            num_masked_indices = torch.bernoulli(torch.full([attention_mask.shape[0], num_count], num_mp))
            cat_masked_indices = torch.bernoulli(torch.full([attention_mask.shape[0], cat_count], cat_mp))
            masked_indices = torch.cat([num_masked_indices, cat_masked_indices], dim=-1)

        # Ensure at least one feature is masked per row
        index_tensor = torch.full(masked_indices.shape, False)
        sum_masked = torch.sum(masked_indices, dim=-1, keepdim=True)
        index_tensor[:, :1] = sum_masked == 0
        masked_indices[index_tensor] = 1
        index_tensor[:, :1] = sum_masked == masked_indices.shape[1]
        masked_indices[index_tensor] = 0

        return masked_indices.int().to(self.device)

    def pretrain_masked(self, X_data, batch_size=128, num_epochs=10, lr=3e-4, shuffle=True, num_workers=0):
        """
        Masked pretraining loop with adaptive masking.
        """
        dummy_y = pd.Series([0] * len(X_data), index=X_data.index)
        trainset = (X_data, dummy_y)

        collator = SupervisedTrainCollator(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            binary_columns=self.binary_columns,
        )

        train_ds = TrainDataset(trainset)
        loader = DataLoader(train_ds, collate_fn=collator, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in loader:
                x_dict, _ = batch
                loss = self.forward_masked(x_dict)
                loss = loss.to(torch.float32)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if epoch%20==0:
                print("Loss: ", avg_loss)

    def forward_masked(self, tokenized_batch):
        """
        1) Process tokenized batch => embeddings
        2) Mask features dynamically
        3) Reconstruct masked values
        4) Compute loss
        """
        outputs, aux_info = self.input_encoder.feature_processor(**tokenized_batch)
        embedding = outputs['embedding'].to(torch.float32).to(self.device)
        attention_mask = outputs['attention_mask'].to(torch.float32).to(self.device)
        num_count = aux_info['num_count']
        col_emb = aux_info['col_emb']
        cat_count = col_emb.shape[0] - num_count
        if num_count>0:
            x_num = aux_info['x_num'].to(torch.float32).to(self.device)
        cat_ref_emb = aux_info['cat_bert_emb']
        masked_indices = self.mask_features(attention_mask, num_count, cat_count)
        masked_emb = self.mask_token(embedding, masked_indices, col_emb)

        encoder_output = self.encoder(embedding=masked_emb, attention_mask=attention_mask)

        num_encoded = encoder_output[:, :num_count, :].to(torch.float32)
        cat_encoded = encoder_output[:, num_count:, :].to(torch.float32)

        num_loss = torch.tensor(0.0, device=self.device).to(torch.float32)
        cat_loss = torch.tensor(0.0, device=self.device).to(torch.float32)

        if num_encoded.nelement() != 0:
            num_encoded = self.num_head(num_encoded)
            num_loss = self.cal_mask_num_features_loss(num_encoded, masked_indices[:, :num_count], x_num)

        if cat_encoded.nelement() != 0:
            cat_encoded = self.cat_head(cat_encoded)
            cat_loss = self.cal_mask_cat_features_loss(cat_encoded, masked_indices[:, num_count:], cat_ref_emb)

        loss = 0.5 * num_loss + cat_loss
        return loss

    def cal_mask_num_features_loss(self, output_emb, masked_indices, x_num):
        """
        Computes loss for numeric feature reconstruction.
        """
        if masked_indices.bool().any():
            output_emb_norm = self._minmax_norm(output_emb)
            x_num = x_num.unsqueeze(dim=-1)
            loss = self.mse_loss(output_emb_norm[masked_indices.bool()], x_num[masked_indices.bool()])
        else:
            loss = torch.tensor(0, device=self.device)

        return loss

    def cal_mask_cat_features_loss(self, output_emb, masked_indices, cat_value_emb):
        """
        Computes loss for categorical feature reconstruction using cosine similarity.
        """
        if masked_indices.bool().any():
            cosine_distance = 1 - F.cosine_similarity(output_emb[masked_indices.bool()], cat_value_emb[masked_indices.bool()], dim=-1)
            loss = torch.mean(cosine_distance)
        else:
            loss = torch.tensor(0, device=self.device)

        return loss

    def _minmax_norm(self, emb, eps=1e-12):
        min_vals, _ = torch.min(emb, dim=0, keepdim=True)
        max_vals, _ = torch.max(emb, dim=0, keepdim=True)
        return (emb - min_vals) / ((max_vals - min_vals) + eps)

























