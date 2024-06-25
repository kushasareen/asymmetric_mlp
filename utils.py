import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data

def linear(x, m = 1, b = 0):
    return m*x + b

def quadratic(x, h= 0, c= 0, a = 1):
    return a*(x - h)**2 + c

def generate_noisy_data(f, datapoints, std, range = [-10, 10], **kwargs):
    t = (range[1] - range[0])*torch.rand((datapoints)) + range[0]
    y = f(t, **kwargs)
        
    return (t, y + std*torch.randn((datapoints)))

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, t, y):
        assert len(t) == len(y)
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, idx):
        return self.t[idx], self.y[idx]

if __name__ == '__main__':
    (t, y) = generate_noisy_data(linear, 100, 2, m = -1, b = 1)
    plt.scatter(t, y)
    # plt.show()

    print(torch.stack((t,y), dim = -1))
