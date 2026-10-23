import torch
import torch.nn.functional as F
import torch, torchaudio


class VAEBaseClass:
    def __init__(self, model, optimizer, data_processor, device):
        self.model = model.to(device)
        self.data_processor = data_processor
        self.optimizer = optimizer
        self.device = device
        self.beta = 0.01

    def loss_function(self, x, recon_x, mean, logvar):
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")

        kl_per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_per_sample.mean()

        # Frequency differences
        recon_f = recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :]
        target_f = x[:, :, 1:, :] - x[:, :, :-1, :]
        freq = F.l1_loss(recon_f, target_f, reduction="mean")
        return recon_loss + 0.1 * freq + self.beta * kl_loss
    
    def train(self, epoch, model, optimizer, data_loader, log_interval=100):
        model.train()
        train_loss = 0
        for batch_idx, (data, *_) in enumerate(data_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = self.loss_function(recon_batch, data, mu, logvar)
        
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

        return train_loss 
    
    def test(self, epoch, model, data_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, *_) in enumerate(data_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 59:
                    self.data_processor.display_reconstruction(data, recon_batch, epoch)
                
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train_test_sample(self, model, optimizer, train_loader, test_loader, epochs, num_samples, latent_dim):
        losses = []
        for epoch in range(1, epochs + 1):
            loss = self.train(epoch, model, optimizer, train_loader)
            losses.append(loss)
            self.test(epoch, model, test_loader)

        return losses
    
    