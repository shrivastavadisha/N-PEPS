import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import params
import math
from model.encoder import DenseEncoder

class BaseModel(nn.Module):
  def load(self, path):
    if use_cuda:
      params = torch.load(path)
    else:
      params = torch.load(path, map_location=lambda storage, loc: storage)

    state = self.state_dict()
    for name, val in params.items():
      if name in state:
        assert state[name].shape == val.shape, "%s size has changed from %s to %s" % \
                             (name, state[name].shape, val.shape)
        state[name].copy_(val)
      else:
        print("WARNING: %s not in model during model loading!" % name)

  def save(self, path):
    torch.save(self.state_dict(), path)

class ScaledDotProductAttention(nn.Module):
  ''' Scaled Dot-Product Attention '''

  def __init__(self, temperature, attn_dropout=0.0): #either apply dropout after adding sc scores or don't apply anything
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q, k, mask=None, PE_solution_scores=None):

    attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #(bs, num_heads, max_len_q, max_len_k)
    #print("attn:", attn.shape)

    if PE_solution_scores is not None:
      sc = PE_solution_scores
      attn = (attn+sc)/2.0

    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)

    attn = self.dropout(F.softmax(attn, dim=-1))#(bs, num_heads, max_len_q, max_len_k)
    return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=4, d_model=params.state_dim, d_k=32, d_v=32,dropout=0.0,
           include_res_ln=True, return_att_weights=False, query_mode='global'):
      super(MultiHeadAttention, self).__init__()

      self.include_res_ln = include_res_ln
      self.query_mode = query_mode
      self.return_att_weights = return_att_weights
      self.n_head = n_head
      self.d_k = d_k
      self.d_v = d_v
      self.d_model = d_model
      self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
      self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
      self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
      self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
      self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
      self.dropout = nn.Dropout(dropout)
      self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, PE_states, global_state, PE_statements, PE_operators, PE_solution_scores,
          att_type, mask):

      d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
      sz_b = global_state.size(0) #batch_size
      q = global_state #(bs, max_len_g, state_dim)
      if self.query_mode == 'PE':
        q = q.view(sz_b, -1, self.d_model)
      k = PE_states.view(sz_b, -1, self.d_model) #(bs, num_examples* max_len_PE, state_dim)
      v_s, v_o = PE_statements.view(sz_b, -1, self.d_model), PE_operators.view(sz_b, -1, self.d_model)#(bs, num_examples* max_len_PE, state_dim)
      len_q, len_k, len_v = q.size(1), k.size(1), v_s.size(1) #(max_len_g, max_len_PE*num_examples, max_len_PE*num_examples)

      residual = q

      # Pass through the pre-attention projection: b x lq x (n*dv)
      # Separate different heads: b x lq x n x dv
      q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
      k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
      v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)
      v_o = self.w_vs(v_o).view(sz_b, len_v, n_head, d_v)

      # Transpose for attention dot product: b x n x lq x dv
      q, k, v_s, v_o = q.transpose(1, 2), k.transpose(1, 2), v_s.transpose(1, 2), v_o.transpose(1, 2)

      if mask is not None:
        mask = mask.unsqueeze(1)   # For head axis broadcasting.

      if att_type =='ca_sc':
        PE_solution_scores = torch.repeat_interleave(PE_solution_scores.view(sz_b, -1).unsqueeze(1), \
                    repeats=len_q, dim=-2).unsqueeze(1)
      else:
        PE_solution_scores = None

      # calculate attention scores
      attn = self.attention(q, k, mask, PE_solution_scores)#(bs, num_heads, max_len_g, max_len_PE)

      statement_pred = torch.matmul(attn, v_s)
      statement_pred = statement_pred.transpose(1, 2).contiguous().view(sz_b, len_q, -1) #(bs, max_len_g, num_heads*state_dim)
      statement_pred = self.dropout(self.fc(statement_pred)) #(bs, max_len_g, state_dim)
      if self.include_res_ln:
        statement_pred += residual
        statement_pred = self.layer_norm(statement_pred)

      operator_pred = torch.matmul(attn, v_o)
      operator_pred = operator_pred.transpose(1, 2).contiguous().view(sz_b, len_q, -1)#(bs, max_len_g, num_heads*state_dim)
      operator_pred = self.dropout(self.fc(operator_pred)) #(bs, max_len_g, state_dim)
      if self.include_res_ln:
        operator_pred += residual
        operator_pred = self.layer_norm(operator_pred)

      if self.return_att_weights:
        return statement_pred, operator_pred, attn
      else:
        return statement_pred, operator_pred, None


    def predict(self, PE_states, global_state, PE_statements, PE_solution_scores, att_type, mask):
      # only statements are predicted here

      d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
      sz_b = global_state.size(0) #batch_size
      q = global_state #(bs, max_len_g, state_dim)
      if self.query_mode == 'PE':
        q = q.view(sz_b, -1, self.d_model)
      k = PE_states.view(sz_b, -1, self.d_model) #(bs, num_examples* max_len_PE, state_dim)
      v_s = PE_statements.view(sz_b, -1, self.d_model)#(bs, num_examples* max_len_PE, state_dim)

      len_q, len_k, len_v = q.size(1), k.size(1), v_s.size(1) #(max_len_g, max_len_PE*num_examples, max_len_PE*num_examples)

      residual = q

      # Pass through the pre-attention projection: b x lq x (n*dv)
      # Separate different heads: b x lq x n x dv
      q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
      k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
      v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

      # Transpose for attention dot product: b x n x lq x dv
      q, k, v_s = q.transpose(1, 2), k.transpose(1, 2), v_s.transpose(1, 2)

      if mask is not None:
        mask = mask.unsqueeze(1)   # For head axis broadcasting.

      if att_type =='ca_sc':
        PE_solution_scores = torch.repeat_interleave(PE_solution_scores.view(sz_b, -1).unsqueeze(1), \
                    repeats=len_q, dim=-2).unsqueeze(1)
      else:
        PE_solution_scores = None

      attn = self.attention(q, k, mask, PE_solution_scores)#(bs, num_heads, max_len_g, max_len_PE)
      statement_pred = torch.matmul(attn, v_s)
      statement_pred = statement_pred.transpose(1, 2).contiguous().view(sz_b, len_q, -1) #(bs, max_len_g, num_heads*state_dim)
      statement_pred = self.dropout(self.fc(statement_pred)) #(bs, max_len_g, state_dim)
      if self.include_res_ln:
        statement_pred += residual
        statement_pred = self.layer_norm(statement_pred)

      if self.return_att_weights:
        return statement_pred, attn
      else:
        return statement_pred, None


class PositionalEncoding(nn.Module):

  def __init__(self, d_hid, n_position=200):
    super(PositionalEncoding, self).__init__()

    # Not a parameter
    self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

  def _get_sinusoid_encoding_table(self, n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
      return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

  def forward(self, x):
    return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionwiseFeedForward(nn.Module):
  ''' A two-feed-forward-layer module '''

  def __init__(self, d_in, d_hid, dropout=0.0):
    super().__init__()
    self.w_1 = nn.Linear(d_in, d_hid) # position-wise
    self.w_2 = nn.Linear(d_hid, d_in) # position-wise
    self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    residual = x
    x = self.w_2(F.relu(self.w_1(x)))
    x = self.dropout(x)
    x += residual
    x = self.layer_norm(x)
    return x

class BasicAggModel(nn.Module):

  def __init__(self, include_value_emb=True, include_pos_emb=True, include_ff=True, include_res_ln=True,
        dropout=0.0, d_inner=2048, d_model=params.state_dim, return_att_weights=False, n_head=8,
        d_k=64, head_params=None, state_emb_params=None, include_state_emb=False, include_dense_proj=True,
        query_mode='global'):
    super(BasicAggModel, self).__init__()

    self.include_value_emb = include_value_emb
    self.query_mode = query_mode
    self.include_pos_emb = include_pos_emb
    self.include_ff = include_ff
    self.include_res_ln = include_res_ln
    self.include_dense_proj = include_dense_proj
    self.include_state_emb = include_state_emb
    self.d_inner = d_inner
    self.d_model = d_model
    self.return_att_weights = return_att_weights
    self.mha = MultiHeadAttention(n_head=n_head, d_k=d_k, d_v=d_k, dropout=dropout,\
                    return_att_weights=return_att_weights, include_res_ln=include_res_ln,
                    query_mode=query_mode)
    self.statement_ff = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=dropout)
    self.operator_ff = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=dropout)
    self.position_enc = PositionalEncoding(d_model)
    self.statement_layer_norm = nn.LayerNorm(params.num_statements, eps=1e-6)
    self.operator_layer_norm = nn.LayerNorm(params.num_operators, eps=1e-6)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(p=dropout)

    if self.include_dense_proj:
      self.statement_dense_proj = nn.Linear(d_model, params.num_statements, bias=False)
      self.operator_dense_proj = nn.Linear(d_model, params.num_operators, bias=False)
    if self.include_value_emb:
      self.statement_emb = nn.Embedding(params.num_statements+1, d_model, padding_idx=params.num_statements)
      self.operator_emb = nn.Embedding(params.num_operators+1, d_model, padding_idx=params.num_operators)

    if self.include_state_emb:
      self.state_encoder = DenseEncoder()

    # parameter initializations
    if head_params!=None and self.include_dense_proj:
      self.statement_dense_proj.weight.data = head_params[0]
      self.operator_dense_proj.weight.data = head_params[1]

    if state_emb_params!=None and self.include_state_emb:
      model_dict = self.state_encoder.state_dict()
      model_dict.update(state_emb_params)
      self.state_encoder.load_state_dict(model_dict)

    for name, p in self.named_parameters():
      if name!='statement_emb.weight' and name!='operator_emb.weight' and name!='statement_dense_proj.weight' \
      and name!='operator_dense_proj.weight' and p.dim() > 1:
        nn.init.xavier_uniform_(p)


  def get_pad_mask(self, seq, pad_idx):
    return (seq != pad_idx)

  def forward(self, PE_states, global_state, PE_statements, PE_operators, PE_solution_scores, att_type):

    if self.include_state_emb:
      max_set_size_PE = PE_states.shape[-3]
      bs = global_state.shape[0]
      max_examples_PE = PE_states.shape[1]

      global_state = self.state_encoder(global_state.view(-1, 5, 12, 22)).view(bs, -1, params.state_dim)
      PE_states = self.state_encoder(PE_states.view(-1, max_set_size_PE, 12, 22)).view(bs, max_examples_PE, -1, params.state_dim)

    #Learn embedding of statements and operators
    if self.include_value_emb:
      PE_statements = self.statement_emb(PE_statements)
      PE_operators = self.operator_emb(PE_operators)

    # calculate mask to be used for attention later
    batch_size = PE_states.size(0)
    q = global_state #(bs, max_len_g, params.state_dim=embedding dim for seq)
    if self.query_mode == 'PE':
      q = q.view(batch_size, -1, self.d_model)
    k = PE_states.view(batch_size, -1, self.d_model) #(bs, max_len_k, params.state_dim)
    attn = torch.matmul(q , k.transpose(1, 2)) #(bs, max_len_g, max_len_k=num_examples*max_len_PE)
    mask = self.get_pad_mask(attn, 0.0) #(bs, max_len_g, max_len_k=num_examples*max_len_PE)

    if self.include_pos_emb:
      num_examples, max_len_PE = PE_states.size(1), PE_states.size(2)
      if self.query_mode == 'PE':
        global_state = global_state.view(-1, max_len_PE, self.d_model)
        global_state = self.dropout(self.position_enc(global_state))#(bs, max_len_g, params.state_dim)
        global_state = global_state.view(-1, num_examples, max_len_PE, self.d_model)
      if self.query_mode =='global':
        global_state = self.dropout(self.position_enc(global_state))#(bs, max_len_g, params.state_dim)
      PE_states = self.dropout(self.position_enc(PE_states.view(-1, max_len_PE, self.d_model)))#(bs*num_examples, max_len_PE, params.state_dim)
      PE_states = PE_states.view(-1, num_examples, max_len_PE, self.d_model)#(bs, num_examples, max_len_PE, params.state_dim)
      PE_statements = self.dropout(self.position_enc(PE_statements.view(-1, max_len_PE, self.d_model)))#(bs*num_examples, max_len_PE, params.state_dim)
      PE_statements = PE_statements.view(-1, num_examples, max_len_PE, self.d_model)#(bs, num_examples, max_len_PE, params.state_dim)
      PE_operators = self.dropout(self.position_enc(PE_operators.view(-1, max_len_PE, self.d_model)))#(bs*num_examples, max_len_PE, params.state_dim)
      PE_operators = PE_operators.view(-1, num_examples, max_len_PE, self.d_model)#(bs, num_examples, max_len_PE, params.state_dim)

    global_state = self.layer_norm(global_state)
    PE_states = self.layer_norm(PE_states)
    PE_statements = self.layer_norm(PE_statements)
    PE_operators = self.layer_norm(PE_operators)

    #Multi-Head Attention
    statement_pred, operator_pred, att_weights = self.mha(PE_states, global_state, PE_statements, PE_operators,
                  PE_solution_scores, att_type, mask) #(bs, max_len_g, state_dim), (bs, max_len_g, state_dim),
                                    #(bs, num_heads, max_len_g, max_len_PE)
    #Positionwise FeedForward
    if self.include_ff:
      statement_pred = self.statement_ff(statement_pred)#(bs, max_len_g, state_dim)
      operator_pred = self.operator_ff(operator_pred)#(bs, max_len_g, state_dim)

    #Final projection to get logits
    if self.include_dense_proj:
      statement_pred = self.statement_dense_proj(statement_pred)#(bs, max_len_g, num_statements)
      operator_pred = self.operator_dense_proj(operator_pred)#(bs, max_len_g, num_operators)

    if self.return_att_weights:
      return statement_pred, operator_pred, att_weights
    else:
      return statement_pred, operator_pred, None

  def predict(self, PE_states, global_state, PE_statements, PE_solution_scores, att_type):

    #Learn embedding of statements and operators
    if self.include_value_emb:
      PE_statements = self.statement_emb(PE_statements)

    # calculate mask to be used for attention later
    batch_size = PE_states.size(0)
    q = global_state #(bs, max_len_g, params.state_dim=embedding dim for seq)
    if self.query_mode == 'PE':
      q = q.view(batch_size, -1, self.d_model)
    k = PE_states.view(batch_size, -1, self.d_model) #(bs, max_len_k=num_examples*max_len_PE, params.state_dim)
    attn = torch.matmul(q , k.transpose(1, 2)) #(bs, max_len_g, max_len_k=num_examples*max_len_PE)
    mask = self.get_pad_mask(attn, 0.0) #(bs, max_len_g, max_len_k=num_examples*max_len_PE)


    if self.include_pos_emb:
      num_examples, max_len_PE = PE_states.size(1), PE_states.size(2)
      if self.query_mode == 'PE':
        global_state = global_state.view(-1, max_len_PE, self.d_model)
        global_state = self.dropout(self.position_enc(global_state))#(bs, max_len_g, params.state_dim)
        global_state = global_state.view(-1, num_examples, max_len_PE, self.d_model)
      if self.query_mode =='global':
        global_state = self.dropout(self.position_enc(global_state))#(bs, max_len_g, params.state_dim)
      PE_states = self.dropout(self.position_enc(PE_states.view(-1, max_len_PE, self.d_model)))#(bs*num_examples, max_len_PE, params.state_dim)
      PE_states = PE_states.view(-1, num_examples, max_len_PE, self.d_model)#(bs, num_examples, max_len_PE, params.state_dim)
      PE_statements = self.dropout(self.position_enc(PE_statements.view(-1, max_len_PE, self.d_model)))#(bs*num_examples, max_len_PE, params.state_dim)
      PE_statements = PE_statements.view(-1, num_examples, max_len_PE, self.d_model)#(bs, num_examples, max_len_PE, params.state_dim)

    global_state = self.layer_norm(global_state)
    PE_states = self.layer_norm(PE_states)
    PE_statements = self.layer_norm(PE_statements)

    statement_pred, att_weights = self.mha.predict(PE_states, global_state, PE_statements, PE_solution_scores,
                    att_type, mask) #(bs, max_len_g, state_dim), (bs, max_len_g, state_dim), (bs, num_heads, max_len_g, max_len_PE)
    #Positionwise FeedForward
    if self.include_ff:
      statement_pred = self.statement_ff(statement_pred)#(bs, max_len_g, state_dim)

    #Final projection to get logits
    if self.include_dense_proj:
      statement_pred = self.statement_dense_proj(statement_pred)#(bs, max_len_g, num_statements)
      statement_pred = F.softmax(statement_pred, dim=-1).data.squeeze(1)

    if self.return_att_weights:
      return statement_pred, att_weights
    else:
      return statement_pred, None

class AttModel(nn.Module):

  def __init__(self, include_pos_emb=True, include_ff=True, include_res_ln=True,
        dropout=0.0, d_inner=2048, d_model=params.state_dim, return_att_weights=False, n_head=8,
        d_k=64, head_params=None, state_emb_params=None, include_self_attention=False, self_attention_type='val',
        include_state_emb=False):
    super(AttModel, self).__init__()

    self.include_self_attention = include_self_attention
    self.self_attention_type = self_attention_type

    if not self.include_self_attention or (self.include_self_attention and self.self_attention_type == 'key'):
      include_value_emb_ca = True
    else:
      include_value_emb_ca = False


    self.return_att_weights = return_att_weights
    self.d_model = d_model

    if self.include_self_attention:
      if self.self_attention_type =='val' or self.self_attention_type=='both':
        self.sa_val = BasicAggModel(include_value_emb=True, include_pos_emb=include_pos_emb, include_ff=include_ff,
          include_res_ln=include_res_ln, dropout=dropout, d_inner=d_inner, d_model=d_model,
          return_att_weights=return_att_weights, n_head=n_head, d_k=d_k, head_params=head_params,
          state_emb_params=state_emb_params, include_state_emb=include_state_emb, include_dense_proj=False,
          query_mode='PE')

      if self.self_attention_type =='key' or self.self_attention_type=='both':
          self.sa_key = BasicAggModel(include_value_emb=False, include_pos_emb=include_pos_emb, include_ff=include_ff,
          include_res_ln=include_res_ln, dropout=dropout, d_inner=d_inner, d_model=d_model,
          return_att_weights=return_att_weights, n_head=n_head, d_k=d_k, head_params=head_params,
          state_emb_params=state_emb_params, include_state_emb=include_state_emb, include_dense_proj=False,
          query_mode='PE')

    self.ca = BasicAggModel(include_value_emb=include_value_emb_ca, include_pos_emb=include_pos_emb,
        include_ff=include_ff, include_res_ln=include_res_ln, dropout=dropout, d_inner=d_inner,
        d_model=d_model, return_att_weights=return_att_weights, n_head=n_head, d_k=d_k, head_params=head_params,
        state_emb_params=state_emb_params, include_state_emb=include_state_emb, include_dense_proj=True,
        query_mode='global')


  def forward(self, PE_states, global_state, PE_statements, PE_operators, PE_solution_scores, att_type):
    #global_state = (bs, max_len_g, params.state_dim=embedding dim for seq)
    #PE_states = (bs, num_of_solutions, max_len_PE, params.state_dim=embedding dim for seq)
    #PE_statements = (bs, num_of_solutions, max_len_PE)

    if self.include_self_attention:
      if self.self_attention_type == 'val' or self.self_attention_type == 'both':
        PE_statements, PE_operators, sa_val_att_weights = self.sa_val(PE_states, PE_states, PE_statements, PE_operators, \
                              PE_solution_scores, att_type)

      if self.self_attention_type == 'key' or self.self_attention_type == 'both':
        num_examples, max_len_PE = PE_states.size(1), PE_states.size(2)
        PE_states, _, sa_key_att_weights = self.sa_key(PE_states, PE_states, PE_states, PE_states, \
                              PE_solution_scores, att_type)
        PE_states = PE_states.view(-1, num_examples, max_len_PE, self.d_model)

    statement_pred, operator_pred, ca_att_weights = self.ca(PE_states, global_state, PE_statements, PE_operators, \
                            PE_solution_scores, att_type)

    if self.return_att_weights:
      return statement_pred, operator_pred, (sa_val_att_weights, sa_key_att_weights, ca_att_weights)
    else:
      return statement_pred, operator_pred, None

  def predict(self, PE_states, global_state, PE_statements, PE_solution_scores, att_type):

    if self.include_self_attention:
      if self.self_attention_type == 'val' or self.self_attention_type == 'both':
        PE_statements, sa_val_att_weights = self.sa_val.predict(PE_states, PE_states, PE_statements, \
                             PE_solution_scores, att_type)

      if self.self_attention_type == 'key' or self.self_attention_type == 'both':
        num_examples, max_len_PE = PE_states.size(1), PE_states.size(2)

        PE_states, sa_key_att_weights = self.sa_key.predict(PE_states, PE_states, PE_states, \
                              PE_solution_scores, att_type)
        PE_states = PE_states.view(-1, num_examples, max_len_PE, self.d_model)

    statement_pred, ca_att_weights = self.ca.predict(PE_states, global_state, PE_statements, \
                            PE_solution_scores, att_type)

    if self.return_att_weights:
      return statement_pred, ca_att_weights
    else:
      return statement_pred, None
