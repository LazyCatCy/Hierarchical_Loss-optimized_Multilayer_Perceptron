from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, classes):
        super().__init__()

        # Linear Layer 1
        self.linear_1 = nn.Linear(input_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Relu 1
        self.relu_1 = nn.ReLU()

        # Linear Layer 2
        self.linear_2 = nn.Linear(output_dim, output_dim)

        # Batch Norm
        self.batch_norm = nn.BatchNorm1d(output_dim)

        # Relu 2
        self.relu_2 = nn.ReLU()

        # Linear Layer 3
        self.linear_3 = nn.Linear(output_dim, classes)

        # ATP Judgement
        self.fc_open = nn.Linear(output_dim, classes * 2, bias=False)

        # Initialize The Model Weights
        self._init_weights()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.batch_norm(x)
        x = self.relu_2(x)
        output = self.linear_3(x)
        probability = self.fc_open(x)
        return output, probability

    def _init_weights(self):
        """Initialize The Model Weights"""
        for m in self.modules():

            # isinstance(object, classinfo): Determine if the classes are the same
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu'
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
