from torch import nn


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class MultimodalModel(nn.Module):
    def __init__(self, anchor_model, posneg_model):
        super(MultimodalModel, self).__init__()
        self.anchor_model = anchor_model
        self.posneg_model = posneg_model

    def forward(self, x, stream):
        if stream == 1:
            return self.anchor_model(x)
        elif stream == 2:
            return self.posneg_model(x)
        else:
            raise ValueError(f'stream should be 1 or 2, not {stream}')

