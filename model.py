from torch import nn

class model(nn.Module):
    def __init__(self,input_shape,units,output_shape):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                input_shape,
                units,
                3,1,1
            ),
            nn.ReLU(),
            nn.Conv2d(
                units,
                units,
                3,1,1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                units,
                units,
                3,1,1
            ),
            nn.ReLU(),
            nn.Conv2d(
                units,
                units,
                3,1,1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(units*7*7,output_shape)
        )

        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.classifier
        )

    def forward(self,x):
        return self.model(x)