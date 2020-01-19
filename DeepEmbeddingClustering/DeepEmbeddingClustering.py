import torchvision
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F 

import numpy as np 
import time 

from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment

from VariationalAutoEncoder import VAE
from Mnist_loader import mnist_data_loader

def acc(y_pred, y_target):
    D = max(y_pred.max(), y_target.max()) + 1
    w = np.zeros( (D, D), dtype=np.int64 )
    for i in range(y_pred.size):
        w[ y_pred[i], y_target[i]] += 1

    ind = linear_assignment( w.max() - w )
    return sum( w[i, j] for i, j in ind) * 1.0 / y_pred.size


def loss_func(feat, cluster_centers, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p

def dist_2_label(q_t):
    _, label = torch.max(q_t, dim=1)
    return label.data.cpu().numpy()

if __name__ == "__main__":
    weights = torch.load('./save_models/pretrain_vae.pkl')
    vae = VAE()
    vae.load_state_dict(weights)

    train_data = torch.load('./mnist/MNIST/processed/training.pt')
    target = train_data[1].numpy()
    train_data_init = Variable(train_data[0].unsqueeze(1).type(torch.FloatTensor) / 255.0)
    feat_init, _ = vae.encode(train_data_init.view(-1, 784))

    kmeans = KMeans(n_clusters=10, n_init=20)
    y_pred_init = kmeans.fit_predict(feat_init.data.cpu().numpy())
    cluster_centers = Variable( (torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)), 
                                requires_grad=True )

    print('Pre-Trained AutoEncoder Accuracy : {}'.format(acc(y_pred_init, target)))

    loss, p = loss_func(feat_init, cluster_centers)

    train_loader, _ = mnist_data_loader()

    optimizer = torch.optim.SGD(list(vae.encoder.parameters()) + list(vae.fc1.parameters()) + [cluster_centers], lr=2.0)

    for epoch in range(20):
        for step, (batch_x, _) in enumerate(train_loader):
            batch_x = Variable(batch_x.view(-1, 784))
            batch_feat, _ = vae.encode(batch_x)
            batch_loss, _ = loss_func(batch_feat, cluster_centers)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if( step % 50 == 0):
                print("Epoch : {}\t step : {}\t batch_loss : {}".format(epoch, step, batch_loss.data.item()))
            
        feat, _ = vae.encode(train_data_init.view(-1, 784))
        loss, p = loss_func(feat, cluster_centers)
        pred_label = dist_2_label(p)
        accuracy = acc(pred_label, target)
        print('=====> Epoch : {}\t Accuracy : {}'.format(epoch, accuracy))
        time.sleep(1)
