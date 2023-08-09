from utils import *
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.morphology import disk, binary_dilation, ball
import open3d as o3d
from ICL_matching import draw_registration_result
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.affine = nn.Linear(3, 3, bias=True)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_pc_as_tensor(mask: np.ndarray, voxel_to_real_space_transformation: Optional[np.ndarray] = None) -> torch.Tensor:
    pc = np.stack(np.where(mask > 0))
    if voxel_to_real_space_transformation is not None:
        pc = affines.apply_affine(voxel_to_real_space_transformation, pc.T).T
    pc = torch.from_numpy(pc).float()
    return pc

def make_reg_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def reg_step(source_pc, target_pc):
        # Sets model to TRAIN mode
        model.train()
        # register
        source_pc_moved = model(source_pc.T)
        # Computes loss

        # todo delete
        # target_pc[:-1, :] = target_pc[:-1, :] / target_pc[-1, :]
        source_pc_moved = source_pc_moved.T
        source_pc_moved[:-1, :] = source_pc_moved[:-1, :].clone() / source_pc_moved[-1, :].clone()
        loss = loss_fn(target_pc[:-1, :], source_pc_moved[:-1, :])

        # todo uncomment
        # loss = loss_fn(target_pc, source_pc_moved.T)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return reg_step


def continuous_dice_for_PCs_loss(radius):

    def loss(pc1: torch.Tensor, pc2: torch.Tensor):
        h = nn.ReLU()(radius - 0.5 * torch.sqrt(((torch.unsqueeze(pc1.T, -1) - torch.unsqueeze(pc2, 0))**2).sum(axis=1)))

        # instead of calculating -d = -h^2(3r -h)/(2(r^2)), it is enough to calculate -d = -h^2(3r -h)
        return ((h ** 3) - 3 * radius * (h ** 2)).sum()

    return loss


def ASSD_loss(pc1: torch.Tensor, pc2: torch.Tensor):
    minus_distances = -torch.sqrt(((torch.unsqueeze(pc1.T, -1) - torch.unsqueeze(pc2, 0))**2).sum(axis=1))
    return -torch.logsumexp(minus_distances, 0).sum() - torch.logsumexp(minus_distances, 1).sum()


def HD_loss(pc1: torch.Tensor, pc2: torch.Tensor):
    minus_distances = -torch.sqrt(((torch.unsqueeze(pc1.T, -1) - torch.unsqueeze(pc2, 0))**2).sum(axis=1))
    pc1_minimum_distances = -torch.logsumexp(minus_distances, 1, keepdim=True)
    pc2_minimum_distances = -torch.logsumexp(minus_distances, 0, keepdim=True)
    pc1_max_of_minimum_distances = torch.logsumexp(pc1_minimum_distances, 0)
    pc2_max_of_minimum_distances = torch.logsumexp(pc2_minimum_distances, 1)
    return torch.logsumexp(torch.cat([pc1_max_of_minimum_distances, pc2_max_of_minimum_distances]), 0)


def get_HD_and_ASSD_loss(ASSD_reg: float = 1, HD_reg: float = 0.3):

    def loss(pc1: torch.Tensor, pc2: torch.Tensor):
        minus_distances = -torch.sqrt(((torch.unsqueeze(pc1.T, -1) - torch.unsqueeze(pc2, 0))**2).sum(axis=1))
        pc1_minimum_distances = -torch.logsumexp(minus_distances, 1, keepdim=True)
        pc2_minimum_distances = -torch.logsumexp(minus_distances, 0, keepdim=True)
        pc1_max_of_minimum_distances = torch.logsumexp(pc1_minimum_distances, 0)
        pc2_max_of_minimum_distances = torch.logsumexp(pc2_minimum_distances, 1)
        assd = (pc1_minimum_distances.sum() + pc2_minimum_distances.sum()) / (minus_distances.size(0) + minus_distances.size(1))
        hd = torch.logsumexp(torch.cat([pc1_max_of_minimum_distances, pc2_max_of_minimum_distances]), 0)
        return ASSD_reg * assd + HD_reg * hd

    return loss


def choose_n_random_points(pc: torch.Tensor, n) -> torch.Tensor:
    perm = torch.randperm(pc.size(1))
    idx = perm[:n]
    return pc[:, idx]


def tensor_to_o3d_vec(tensor_pc):
    np_pc = tensor_pc.cpu().numpy()
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np_pc.T)
    return o3d_pc


def get_affine_transform_from_linear_layer(T):
    # todo uncomment
    # aff = np.eye(4)
    # with torch.no_grad():
    #     aff[:3, :3] = T.weight.cpu().numpy()
    #     aff[:3, 3] = T.bias.cpu().numpy()
    # return aff

    # todo delete
    with torch.no_grad():
        aff = T.weight.cpu().numpy()
    return aff

