import torch
from torch.autograd import Variable

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU()
        )

        self.fc1 = torch.nn.Linear(200, 10)
        self.fc2 = torch.nn.Linear(200, 10)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 784),
            torch.nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar
    
    # torch.exp(input, out=None) → Tensor
    # torch.mul()
    # torch.Tensor.normal_() - in-place version of torch.normal()
    # noise_like_grad = X.data.new(X.size()).normal_(0,0.01)
    # This will create a tensor of Gaussian noise, the same shape and data type as a Variable X:
    def reparameterize(self, mu, logvar):
        # std = torch.exp(torch.mul(logvar, 0.5)) 
        # eps = Variable(std.data.new(std.size()).normal_())
        # z = eps.mul(std).add(mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return rercon_x, mu, logvar
