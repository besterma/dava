import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import logsumexp
import gin.tf


@gin.configurable(
    "pipe",
    blacklist=["ground_truth_data", "encode", "decode", "reconstruct", "random_state",
               "artifact_dir"])
def compute_pipe(ground_truth_data,
                 encode,
                 decode,
                 reconstruct,
                 random_state,
                 artifact_dir=None,
                 num_samples=50000,
                 batch_size=32,
                 z_dim=10,
                 uniform=True,
                 num_runs=1):
    """Computes the stability metric

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    encode: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    decode: Function that takes latent representations as input and outputs observations
    reconstruct: combination of encode + decode
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_samples: How many samples to generate from model
    batch_size: Batch size for sampling.
    z_dim: size of latent space of model
    uniform: sampling strategy to use for factorized posterior sampling
    num_runs:

  Returns:
    Dict with PIPE metric and additional measurements.
  """
    print("Compute PIPE metric")
    score_dict = {}
    print(f"Working on {num_samples}")
    bs = int(np.min((num_samples // 15, batch_size)))
    accuracies, epoch_accuracies, kl_divs, n_active, tc = pipe_overall_accuracy(bs, decode, encode,
                                                                                ground_truth_data, num_samples,
                                                                                random_state, reconstruct,
                                                                                z_dim,
                                                                                uniform, num_runs)
    score_dict[f"score.final_score"] = 2 * (1 - epoch_accuracies[0][9])  # This is the number of samples we finally used
    score_dict[f"epoch_accuracies.{num_samples}"] = epoch_accuracies[0]
    score_dict[f"kl_divs.{num_samples}"] = str(kl_divs)
    score_dict[f"n_active.{num_samples}"] = n_active
    score_dict[f"tc.{num_samples}"] = tc

    del artifact_dir
    return score_dict


def pipe_overall_accuracy(batch_size, decode, encode, ground_truth_data, num_samples, random_state,
                          reconstruct, z_dim, uniform, num_runs):
    def compute_gaussian_kl(z_mean, z_logvar):
        return np.mean(
            0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1),
            axis=0)

    N = num_samples - (num_samples % batch_size)
    class_one_images, class_two_images, kl_divs, n_active, tc = generate_reconstructed_sampled(N, batch_size,
                                                                                               compute_gaussian_kl,
                                                                                               decode,
                                                                                               encode,
                                                                                               ground_truth_data,
                                                                                               random_state,
                                                                                               reconstruct, z_dim,
                                                                                               uniform=uniform)

    images, labels = aggregate_and_get_labels(class_one_images, class_two_images)
    num_channels = images.shape[-1]
    if num_channels != 1 and num_channels != 3:
        # this might happen with pytorch data channel ordering
        num_channels = images.shape[-3]
    accuracies = []
    epoch_accuracies = []
    for _ in range(num_runs):
        print("split into appropriately sized train/test")
        x_train, x_test, y_train, y_test = train_test_split(images, labels,
                                                            test_size=0.5,
                                                            stratify=labels,
                                                            random_state=random_state.randint(0, 100000),
                                                            shuffle=True)
        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train,
                                                            test_size=0.1,
                                                            stratify=y_train,
                                                            random_state=random_state.randint(0, 100000),
                                                            shuffle=True)
        train_dataset = generate_dataset_from_numpy(x_train, y_train)
        del x_train, y_train
        eval_dataset = generate_dataset_from_numpy(x_eval, y_eval)
        del x_eval, y_eval
        test_dataset = generate_dataset_from_numpy(x_test, y_test)
        del x_test, y_test
        accuracy, epoch_accuracy = pipe_accuracy(train_dataset, eval_dataset, test_dataset,
                                                 num_channels, batch_size, random_state)
        accuracies.append(accuracy)
        epoch_accuracies.append(epoch_accuracy)
    print(f"Individual accuracies: {accuracies}")

    return accuracies, epoch_accuracies, kl_divs, n_active, tc


@torch.enable_grad()
def pipe_accuracy(train_dataset, eval_dataset, test_dataset, num_channels, batch_size, random_state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state.randint(0, 100000))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state.randint(0, 100000))
    model = ConvDiscriminator(1, num_channels)
    model.to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print("train discriminator")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            pin_memory=torch.cuda.is_available())
    best_accuracy = 0
    best_model_state_dict = model.state_dict()
    epoch_accuracies = []
    current_step = 0
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            current_step += batch_size
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if current_step >= 5000:
                current_step = 0
                accuracy = compute_accuracy(batch_size, model, test_dataset)
                epoch_accuracies.append(accuracy)

        eval_accuracy = compute_accuracy(batch_size, model, eval_dataset)
        print('[%d, %5d] loss: %.3f, accuracy: %.2f' %
              (epoch + 1, i + 1, running_loss / 2000, eval_accuracy))
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_model_state_dict = model.state_dict()
    print("test discriminator on test set")
    model.load_state_dict(best_model_state_dict)
    accuracy = compute_accuracy(batch_size, model, test_dataset)
    model.to("cpu")
    print(f"Test accuracy {accuracy}")
    return accuracy, epoch_accuracies


def compute_accuracy(batch_size, model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_total = len(test_dataset)
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred_y = outputs >= 0.5
        num_correct += torch.sum(pred_y == labels).detach().to("cpu").numpy()
    accuracy = (num_correct * 100.0 / num_total)

    return accuracy


def aggregate_and_get_labels(reconstructed_images, sampled_images):
    reconstructed_images = np.vstack(np.array(reconstructed_images))
    sampled_images = np.vstack(np.array(sampled_images))
    all_images = np.concatenate((reconstructed_images, sampled_images))
    labels = np.hstack((np.zeros(reconstructed_images.shape[0], dtype=np.uint8),
                        np.ones(sampled_images.shape[0], dtype=np.uint8)))
    labels = np.expand_dims(labels, 1)
    return all_images, labels


def generate_dataset_from_numpy(images, labels):
    tensor_x = torch.Tensor(images)
    tensor_y = torch.Tensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def generate_reconstructed_sampled(N, batch_size, compute_gaussian_kl, decode, encode, ground_truth_data, random_state,
                                   reconstruct, z_dim, pytorch=False, uniform=True):
    nr_batches = int(N / batch_size)
    means = None
    logvars = None
    sampled = None
    tcs = np.zeros(nr_batches)
    reconstructed_images = []
    print("get dataset of reconstructed images")
    for i in range(nr_batches):
        if pytorch:
            indices = random_state.randint(len(ground_truth_data), size=batch_size)
            images = np.expand_dims(ground_truth_data[indices], axis=3)
        else:
            images = ground_truth_data.sample_observations(batch_size, random_state)
        mean, logvar = encode(images)
        if means is None:
            z_dim = mean.shape[1]
            means = np.zeros((N, z_dim))
            logvars = np.zeros((N, z_dim))
            sampled = np.zeros((N, z_dim))
        reconstructed = reconstruct(images)
        means[i * batch_size: (i + 1) * batch_size] = mean
        logvars[i * batch_size: (i + 1) * batch_size] = logvar
        sample = mean + np.exp(logvar / 2) * np.random.normal(0, 1, logvar.shape)
        tcs[i] = total_correlation(sample, mean, logvar, N)
        sampled[i * batch_size: (i + 1) * batch_size] = sample
        reconstructed_images.append(reconstructed)
    kl_divs = compute_gaussian_kl(means, logvars)
    tc = tcs.mean()
    active = [i for i in range(z_dim) if kl_divs[i] > 0.01]
    inactive = [i for i in range(z_dim) if kl_divs[i] <= 0.01]
    n_active = len(active)
    print("get dataset of sampled images")
    if uniform:
        print("sample uniform")
        random_code = (np.max(means, axis=0) - np.min(means, axis=0)) * random_state.random_sample((N, z_dim)) + np.min(
            means,
            axis=0)
    else:
        print("sample aggregated posterior")
        random_code = np.zeros((N, z_dim))
        for j in range(z_dim):
            indices = random_state.permutation(N)
            random_code[:, j] = means[indices, j]

    sampled_images = []
    for i in range(nr_batches):
        images = decode(random_code[i * batch_size: (i + 1) * batch_size])
        sampled_images.append(images)
    return reconstructed_images, sampled_images, kl_divs, n_active, tc


def gaussian_log_density(z_sampled,
                         z_mean,
                         z_logvar):
    normalization = np.log(2. * np.pi)
    inv_sigma = np.exp(-z_logvar)
    tmp = (z_sampled - z_mean)
    return -0.5 * (tmp * tmp * inv_sigma + z_logvar + normalization)


def total_correlation(z,
                      z_mean,
                      z_logvar,
                      dataset_size: int):
    batch_size = z.shape[0]
    num_latents = z.shape[1]
    constant = (num_latents - 1) * np.log(batch_size * dataset_size)
    log_qz_prob = gaussian_log_density(np.expand_dims(z, 1), np.expand_dims(z_mean, 0),
                                       np.expand_dims(z_logvar, 0))

    log_qz_product = np.sum(
        logsumexp(log_qz_prob, axis=1) + constant,
        axis=1
    )
    log_qz = logsumexp(
        np.sum(log_qz_prob, axis=2),
        axis=1
    ) + constant
    return np.mean(log_qz - log_qz_product)


class ConvDiscriminator(nn.Module):
    def __init__(self, output_dim, num_channels):
        super(ConvDiscriminator, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.output_act = nn.Sigmoid()

        self.conv1 = nn.Conv2d(num_channels, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, self.num_channels, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return self.output_act(z)

