import torch
import torch.nn.functional as F
import torch, torchaudio


class VAEBaseClass:
    def __init__(self, model, optimizer, data_processor, device):
        self.model = model.to(device)
        self.data_processor = data_processor
        self.optimizer = optimizer
        self.device = device
        self.beta = 0.001

    def loss_function(self, x, recon_x, mean, logvar):
        recon = F.l1_loss(recon_x, x)
        kl = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)).mean()
        freq = F.l1_loss(recon_x[:, :, 1:] - recon_x[:, :, :-1],
                        x[:, :, 1:] - x[:, :, :-1])
        self.loss_terms = (recon.item(), freq.item(), kl.item())
        return recon + 100 * freq + self.beta * kl
    
    def train(self, epoch, model, optimizer, data_loader, log_interval=100):
        model.train()
        train_loss = 0
        n_batches = 0
        terms = torch.zeros(3)  # recon, freq, kl

        for batch_idx, (data, *_) in enumerate(data_loader):
            data = data.to(self.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = self.loss_function(data, recon_batch, mu, logvar)  # x first, then recon_x
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            terms += torch.tensor(self.loss_terms)
            n_batches += 1

        avg_loss = train_loss / n_batches
        avg_terms = terms / n_batches
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} | '
            f'recon: {avg_terms[0]:.4f}  freq: {avg_terms[1]:.4f}  KL: {avg_terms[2]:.2f}')
        return avg_loss, avg_terms
    
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
        loss_terms = []
        for epoch in range(1, epochs + 1):
            loss, loss_term = self.train(epoch, model, optimizer, train_loader)
            losses.append(loss)
            loss_terms.append(loss_term)
            self.test(epoch, model, test_loader)

        return losses, loss_terms
    
    