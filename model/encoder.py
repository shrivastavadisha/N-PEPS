import torch
import torch.nn.functional as F
from torch import nn

import params

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.selu(self.linear(x))


class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(torch.cat(inputs, dim=-1))
            inputs.append(output)
        return inputs[-1]


class DenseEncoder(nn.Module):
    def __init__(self):
        super(DenseEncoder, self).__init__()

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.dense = DenseBlock(10, params.dense_growth_size, params.var_encoder_size * params.state_len,
                                params.dense_output_size)

    def forward(self, x):
        #print("x1:", x.shape)
        x, num_batches, num_examples = self.embed_state(x)
        #print("x2:", x.shape)
        x = F.selu(self.var_encoder(x))
        #print("x3:", x.shape)
        x = x.view(num_batches, num_examples, -1)
        #print("x4:",x.shape)
        x = self.dense(x)
        #print("x5:", x.shape)
        x = x.mean(dim=1)
        #print("x6:", x.shape)
        return x.view(num_batches, -1)

    def embed_state(self, x):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        #assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]
        num_examples = x.size()[1]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches, num_examples
