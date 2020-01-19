# 
import torchvision
from torchvision.utils import save_image 
from Mnist_loader import mnist_data_loader

import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim.lr_scheduler as lr_scheduler
"""
optimizer는 step() method를 통해 argument로 전달받은 parameter를 업데이트한다.
모델의 parameter별로(per-parameter) 다른 기준(learning rate 등)을 적용시킬 수 있다. 참고
torch.optim.Optimizer(params, defaults)는 모든 optimizer의 base class이다.
nn.Module과 같이 state_dict()와 load_state_dict()를 지원하여 optimizer의 상태를 저장하고 불러올 수 있다.
zero_grad() method는 optimizer에 연결된 parameter들의 gradient를 0으로 만든다.
torch.optim.lr_scheduler는 epoch에 따라 learning rate를 조절할 수 있다.
"""

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
        std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = Variable(std.data.new(std.size()).normal_())
        # z = eps.mul(std).add(mu)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_func(x, recon_x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    kld = (-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= 784 * 128
    return bce + kld


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, label_train) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        loss = loss_func(data, recon_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} {:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset), 
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            ))
    
    avg_loss = train_loss / len(train_loader.dataset) 
    print('===> Epoch: {} Average Loss: {:.4f}'.format(
        epoch,
        avg_loss
    ))
    return avg_loss 

def test(epoch):
    vae.eval()
    test_loss = 0
    with torch.no_grad(): # volatile(True) == requirer_grad(False)
        for i, (data, lb) in enumerate(test_loader):
            data = Variable(data) 
            recon_batch, mu, logvar = vae(data)
            test_loss += loss_func(data, recon_batch, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(128, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(), 'pretrain_vae/reconstruction_' + str(epoch) + ".png", nrow=n)
        
        test_loss /= len(test_loader.dataset) 
        print('==================> Test Set Loss : {:.4f}'.format(test_loss))
    return test_loss


if __name__ == "__main__":
    train_loader, test_loader = mnist_data_loader()
    vae = VAE()
    optimizer = torch.optim.Adam(params=vae.parameters(), 
                                    lr=1e-3)
    # torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                mode='min')

    for epoch in range(20):
        train(epoch)
        val_loss = test(epoch)
        scheduler.step(val_loss)
        torch.save(vae.state_dict(), 'save_models/pretrain_vae.pkl')
