import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy.typing as npt

from dava.models import BetaVAE, FVAEDiscriminator
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=150000)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--store_path", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--z_dim", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=1000)
    parser.add_argument("--batch_norm", action='store_true')
    parser.add_argument("--spectral_norm", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--instance_norm", action='store_true')
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--annealed_loss_degree", type=int, default=0)

    args = parser.parse_args()

    init_random_state(args.random_seed)
    dataset_path = args.dataset_path
    num_channels = args.num_channels
    print("Loading Dataset")
    dataset = np.load(dataset_path)["images"]
    print("Finished loading Dataset")
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_steps = args.num_steps * batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    store_path = args.store_path
    gamma = args.gamma
    use_batch_norm = args.batch_norm
    use_spectral_norm = args.spectral_norm
    use_layer_norm = args.layer_norm
    use_instance_norm = args.instance_norm
    max_grad_norm = args.max_grad_norm
    prepare_store_path(store_path, dataset_path, num_channels, z_dim)
    store_dict(args.__dict__, os.path.join(store_path, "parameters.txt"))

    vae = BetaVAE(z_dim=z_dim, num_channels=num_channels, beta=gamma,
                  use_batch_norm=use_batch_norm, use_spectral_norm=use_spectral_norm,
                  use_layer_norm=use_layer_norm, use_instance_norm=use_instance_norm).to(device)
    disc = FVAEDiscriminator(z_dim=z_dim).to(device)

    train_factor_vae(vae, disc, gamma, dataset, device, store_path=store_path, batch_size=batch_size,
                     num_steps=num_steps, learning_rate=learning_rate,  max_grad_norm=max_grad_norm)


def train_factor_vae(vae: BetaVAE, disc: FVAEDiscriminator, gamma: int, data: npt.ArrayLike, device, store_path: str,
                     batch_size=32, num_steps=5, learning_rate=1e-4, max_grad_norm=1.):
    assert vae.beta == 1, "FactorVAE beta must be one"
    print("Start FactorVAE training")
    data_loader = DataLoader(data, batch_size=batch_size * 2, pin_memory=True,
                             drop_last=True, shuffle=True, num_workers=0)
    opt_enc = optim.Adam(vae.encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    opt_dec = optim.Adam(vae.decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-08)

    loss_disc = nn.CrossEntropyLoss(reduction='mean')

    running_loss_vae = 0.0
    current_step = 0

    end_reached = False

    rec_label = 0
    sampled_label = 1
    label_rec = torch.full((batch_size,), rec_label, dtype=torch.long, device=device)
    label_sampled = torch.full((batch_size,), sampled_label, dtype=torch.long, device=device)

    while not end_reached:
        for i, x in enumerate(data_loader):
            current_step += batch_size
            if current_step > num_steps:
                end_reached = True
                break

            vae.train()
            x = x.to(device)
            x = x.type(torch.float32) / 255.
            # Have a separate batch for vae update and disc update
            x_vae = x[:batch_size]
            x_disc = x[batch_size:]

            vae.zero_grad()
            disc.zero_grad()

            # train vae with reconstruction + tc_loss
            _, z_t, vae_loss = vae(x_vae, current_step)

            disc_output = disc(z_t)
            vae_tc_loss = (disc_output[:, :1] - disc_output[:, 1:]).mean()
            vae_loss += gamma * vae_tc_loss

            vae_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=max_grad_norm)
            opt_enc.step()
            opt_dec.step()
            running_loss_vae += vae_loss.detach().mean()

            # train discriminator
            _, z_t, _ = vae(x_disc, current_step)
            z_hat = torch.zeros((batch_size, vae.z_dim), device=device)
            for j in range(vae.z_dim):
                indices = torch.randperm(batch_size)
                z_hat[:, j] = z_t[indices, j].detach()
            disc.zero_grad()
            disc_output_p = disc(z_hat)
            disc_loss = 0.5 * (loss_disc(disc_output, label_rec) + loss_disc(disc_output_p, label_sampled))
            disc_loss.backward(inputs=list(disc.parameters()))
            opt_disc.step()

        save_checkpoint(vae, path=store_path)

    print("Finished Training")
    save_checkpoint(vae, path=store_path)


if __name__ == '__main__':
    main()
