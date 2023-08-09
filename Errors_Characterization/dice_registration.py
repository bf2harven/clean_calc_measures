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


if __name__ == '__main__':

    debug = True
    show_impact_on_tumors = True
    load_improved_bl = False
    use_tumors = False

    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # case_name = 'BL_G_G_17_06_2020_FU_G_G_16_08_2020'
    # case_name = 'BL_A_Y_22_01_2020_FU_A_Y_04_05_2020'
    case_name = 'BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020'
    pair_dir = f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/{case_name}'
    if load_improved_bl:
        bl_liver_file_path = f'{pair_dir}/improved_registration_BL_Scan_Liver.nii.gz'
    else:
        bl_liver_file_path = f'{pair_dir}/BL_Scan_Liver.nii.gz'
    fu_liver_file_path = f'{pair_dir}/FU_Scan_Liver.nii.gz'

    bl_liver, file = load_nifti_data(bl_liver_file_path)
    fu_liver, _ = load_nifti_data(fu_liver_file_path)

    selem = disk(1).reshape([3, 3, 1])
    # selem = ball(1)
    bl_liver_border = np.logical_xor(binary_dilation(bl_liver, selem), bl_liver)
    fu_liver_border = np.logical_xor(binary_dilation(fu_liver, selem), fu_liver)

    # bl_liver_border_pc = get_pc_as_tensor(bl_liver_border, file.affine).to(device)
    # fu_liver_border_pc = get_pc_as_tensor(fu_liver_border, file.affine).to(device)
    bl_liver_border_pc = get_pc_as_tensor(bl_liver_border, file.affine)
    fu_liver_border_pc = get_pc_as_tensor(fu_liver_border, file.affine)

    bl_working_pc = bl_liver_border_pc
    fu_working_pc = fu_liver_border_pc

    bl_liver_border_pc_o3d = tensor_to_o3d_vec(bl_liver_border_pc)
    fu_liver_border_pc_o3d = tensor_to_o3d_vec(fu_liver_border_pc)

    if show_impact_on_tumors or use_tumors:
        if load_improved_bl:
            bl_tumors_file_path = glob(f'{pair_dir}/improved_registration_BL_Scan_Tumors_unique_*')[0]
        else:
            bl_tumors_file_path = glob(f'{pair_dir}/BL_Scan_Tumors_unique_*')[0]
        fu_tumors_file_path = glob(f'{pair_dir}/FU_Scan_Tumors_unique_*')[0]

        if use_tumors:
            bl_tumors_pc = get_pc_as_tensor(load_nifti_data(bl_tumors_file_path)[0], file.affine)
            fu_tumors_pc = get_pc_as_tensor(load_nifti_data(fu_tumors_file_path)[0], file.affine)

            bl_tumors_pc_o3d = tensor_to_o3d_vec(bl_tumors_pc)
            fu_tumors_pc_o3d = tensor_to_o3d_vec(fu_tumors_pc)

            bl_working_pc = torch.cat([bl_working_pc, bl_tumors_pc], 1)
            fu_working_pc = torch.cat([fu_working_pc, fu_tumors_pc], 1)

        else:
            bl_tumors_pc_o3d = o3d.geometry.PointCloud()
            bl_tumors_pc_o3d.points = o3d.utility.Vector3dVector(affines.apply_affine(file.affine, np.stack(np.where(load_nifti_data(bl_tumors_file_path)[0] > 0)).T))

            fu_tumors_pc_o3d = o3d.geometry.PointCloud()
            fu_tumors_pc_o3d.points = o3d.utility.Vector3dVector(affines.apply_affine(file.affine, np.stack(np.where(load_nifti_data(fu_tumors_file_path)[0] > 0)).T))


    # hyper-parameters
    batch_size = 5000
    lr = 1e-7
    n_epochs = 200
    radius = 2
    epsilon = 1e-5
    wd = 1

    def get_random_hyper_parameters():

        batch_size_s = [5000, 7000, 10000]
        lr_s = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        radius_s = [1, 2, 3]
        wd_s = [1, 0.97, 0.95, 0.9]

        batch_size = np.random.choice(batch_size_s, 1)[0]
        lr = np.random.choice(lr_s, 1)[0]
        radius = np.random.choice(radius_s, 1)[0]
        wd = np.random.choice(wd_s, 1)[0]

        return batch_size, lr, radius, wd

    n_iterations = 1

    results = {}

    for i in range(n_iterations):

        # batch_size, lr, radius, wd = get_random_hyper_parameters()
        batch_size, lr, radius, wd = (5000, 1e-5, 15, 0)

        # todo uncomment
        # T = nn.Linear(3, 3).to(device)
        # T.weight.data.copy_(torch.eye(3) + epsilon)
        # T.bias.data.copy_(torch.zeros(3) + epsilon)

        # todo delete
        bl_working_pc = torch.cat([bl_working_pc, torch.ones((1, bl_working_pc.size(1)))])
        fu_working_pc = torch.cat([fu_working_pc, torch.ones((1, fu_working_pc.size(1)))])
        T = nn.Linear(4, 4, bias=False).to(device)
        T.weight.data.copy_(torch.eye(4) + epsilon)

        print(f'---------------------------------- batch_size={batch_size}, lr={lr}, radius={radius}, wd={wd} ----------------------------------')

        np.set_printoptions(suppress=True)
        aff = get_affine_transform_from_linear_layer(T)
        if debug:
            print(aff, end='\n------------------------\n')
            draw_registration_result(bl_liver_border_pc_o3d, fu_liver_border_pc_o3d, aff, 'Original - Liver', (5, 1))
            if show_impact_on_tumors:
                draw_registration_result(bl_tumors_pc_o3d, fu_tumors_pc_o3d, aff, 'Original - Tumors', (5, 1))

        # loss_fn = continuous_dice_for_PCs_loss(radius)
        loss_fn = get_HD_and_ASSD_loss(ASSD_reg=2, HD_reg=0.5)
        optimizer = optim.SGD(T.parameters(), lr=lr, weight_decay=wd)

        # Creates the train_step function for our model, loss function and optimizer
        reg_step = make_reg_step(T, loss_fn, optimizer)
        losses = []
        best_weights = aff
        prev_weights = aff
        best_loss = np.inf

        # For each epoch...
        for epoch in range(n_epochs):
            # Performs one registration step and returns the corresponding loss
            bl_points = choose_n_random_points(bl_working_pc, batch_size).to(device)
            fu_points = choose_n_random_points(fu_working_pc, batch_size).to(device)
            # draw_registration_result(tensor_to_o3d_vec(bl_points), tensor_to_o3d_vec(fu_points), window_name='Current points')
            loss = reg_step(bl_points, fu_points)
            # losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_weights = prev_weights
            losses.append(loss)
            prev_weights = get_affine_transform_from_linear_layer(T)

            print(f'epoch: {epoch}, loss: {loss}')
            if loss == 0 or torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                break

        # print(losses)
        # aff = get_affine_transform_from_linear_layer(T)
        print('\nBest parameters:')
        print(best_weights, end='\n------------------------\n')
        plt.plot(losses)
        plt.show()
        # draw_registration_result(bl_liver_border_pc_o3d, fu_liver_border_pc_o3d, aff, f'After {epoch + 1} epochs', (5, 1))

        results[(batch_size, lr, radius, wd)] = (losses, best_loss, best_weights)
        if debug:
            draw_registration_result(bl_liver_border_pc_o3d, fu_liver_border_pc_o3d, best_weights, f'Best aff - Liver', (5, 1))
            if show_impact_on_tumors:
                draw_registration_result(bl_tumors_pc_o3d, fu_tumors_pc_o3d, best_weights, 'Best aff - Tumors', (5, 1))

    r = {}
    for k in results:
        if len(results[k][0]) == n_epochs:
            r[k] = results[k]

    print('')
