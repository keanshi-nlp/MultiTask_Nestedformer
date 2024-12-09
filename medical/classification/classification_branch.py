import sys
sys.path.append("..")
from model.encoder.global_poolformer import Encoder

class MLP(nn.module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Classifier(nn.Module):
    def __init__(self, image_size, fea, pool_size):

        super().__init__()
        self.pool_size = pool_size
        image_size_s = [image_size]

        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]
            image_size_s.append((image_size_s[-1][0] // p[0], image_size_s[-1][1] // p[1], image_size_s[-1][2] // p[2]))

        self.encoder = Encoder(model_num=1, 
                               img_size=image_size_s[1:], 
                               fea=fea, pool_size=pool_size)
        
        self.mlp = MLP(input_size=image_size, 
                       hidden_size=image_size * 2, 
                       output_size=image_size)

    def forward(self, x, adc):

        img = self.encoder(adc)
        img = torch.cat((x, img), 0)
        img = torch.flatten(img, 1)
        out = self.mlp(img)

        return out