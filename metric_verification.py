# Run plot fucntion before running this script
# python3 metric_verification.py --pretrained log_OCTA --task OCTA

import os
import sys
import torch
from torch.autograd import grad
import numpy as np
import argparse
import importlib
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

parser = argparse.ArgumentParser(description="")
parser.add_argument('--pretrained', type=str)
parser.add_argument('--task', type=str, help='Name of the model.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Disable cuda.')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--num_train', type=int, default=131072, help='Number of samples for training.') # 4096 * 32
parser.add_argument('--num_test', type=int, default=32768, help='Number of samples for testing.') # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr_step', type=int, default=5, help='')
parser.add_argument('--lambda', type=float, dest='_lambda', default=0.5, help='Convergence rate: lambda')
parser.add_argument('--w_ub', type=float, default=10, help='Upper bound of the eigenvalue of the dual metric.')
parser.add_argument('--w_lb', type=float, default=0.1, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--log', type=str, help='Path to a directory for storing the log.')
args = parser.parse_args()


effective_dim_start = 4
effective_dim_end = 12
system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control


model = importlib.import_module('model_'+args.task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb)

# Load the model state from the checkpoint
model_path = args.pretrained + '/model_best.pth.tar'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Access the items
loss, p1, p2, l3, c4 = checkpoint['precs']
model_W_state = checkpoint['model_W']
model_Wbot_state = checkpoint['model_Wbot']
model_u_w1_state = checkpoint['model_u_w1']
model_u_w2_state = checkpoint['model_u_w2']

# To load the state dicts into models:
model_W.load_state_dict(model_W_state)
model_Wbot.load_state_dict(model_Wbot_state)
model_u_w1.load_state_dict(model_u_w1_state)
model_u_w2.load_state_dict(model_u_w2_state)

# Load the controller
controller_path = args.pretrained + '/controller_best.pth.tar'
_controller = torch.load(controller_path, map_location=torch.device('cpu'))
_controller.cpu()

# Bbot function
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func
if 'Bbot_func' not in locals():
    def Bbot_func(x): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        # if args.use_cuda:
        #     Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)    


def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum()  # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)
    

# Load the best path in numpy arrays
data = np.load(args.pretrained + '/best_path.npz')
x_closed = data['x_closed']
xstar = data['xstar']
ustar = data['ustar']
n_traj = x_closed.shape[0]  # number of trajectories

for testing_traj in range(n_traj):
    print(f"Testing trajectory {testing_traj+1}/{n_traj}...")
    x = torch.tensor(x_closed[testing_traj, 1:, :, :], dtype=torch.float32)  # x: bs x n x 1
    x = x.requires_grad_()
    xref = torch.tensor(xstar[1:, :, :], dtype=torch.float32)  # xref: bs x n x 1
    uref = torch.tensor(ustar, dtype=torch.float32)  # uref: bs x m x 1
    bs = x.shape[0]

    def forward(x, xref, uref, _lambda=args._lambda, verbose=False, acc=False, detach=False):
        # x: bs x n x 1
        bs = x.shape[0]
        W = W_func(x)
        M = torch.inverse(W)
        f = f_func(x)
        B = B_func(x)
        DfDx = Jacobian(f, x)
        DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
        for i in range(num_dim_control):
            DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)

        _Bbot = Bbot_func(x)  # Bbot: bs x n x (n-m)
        u = u_func(x, x - xref, uref)  # u: bs x m x 1 # TODO: x - xref
        K = Jacobian(u, x)

        A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])
        dot_x = f + B.matmul(u)
        dot_M = weighted_gradients(M, dot_x, x, detach=detach)  # DMDt
        if detach:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
        else:
            Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * _lambda * M 

        # C1
        C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + 2 * _lambda * W
        C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot) # this has to be a negative definite matrix

        # C2
        C2_inners = []
        C2s = []
        for j in range(num_dim_control):
            C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
            C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)
        
        return Contraction, C1_LHS_1, C2s

    Contraction, C1_LHS_1, C2s = forward(x, xref, uref, _lambda=args._lambda, verbose=False, acc=False, detach=False)
    
    # Contraction: bs x n x n
    try:
        eigvals = torch.linalg.eigvalsh(Contraction, UPLO='L')
    except RuntimeError as e:
        print("eigvalsh failed:", e)
        continue
    analyse_contraction = ((torch.linalg.eigvalsh(Contraction, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy()
    print(f"Contraction analysis: {analyse_contraction.mean()}")
    num_pos_eigen_contraction = ((torch.linalg.eigvalsh(Contraction, UPLO='L') >= 0).sum(dim=1)).cpu().detach().numpy()
    # print(f"Number of positive eigenvalues in contraction: {num_pos_eigen_contraction}")

    # C1_LHS_1: bs x (n-m) x (n-m)
    try:
        eigvals_C1 = torch.linalg.eigvalsh(C1_LHS_1, UPLO='L')
    except RuntimeError as e:
        print("eigvalsh C1 failed:", e)
        continue
    analyse_C1 = ((torch.linalg.eigvalsh(C1_LHS_1, UPLO='L') >= 0).sum(dim=1) == 0).cpu().detach().numpy()
    print(f"C1 analysis: {analyse_C1.mean()}")
    num_pos_eigen_C1 = ((torch.linalg.eigvalsh(C1_LHS_1, UPLO='L') >= 0).sum(dim=1)).cpu().detach().numpy()
    # print(f"Number of positive eigenvalues in C1: {num_pos_eigen_C1}")

    # C2s: list of bs x (n-m) x (n-m)
    print(f"Sum of squared C2s: {sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item()}")
    print('\n')
    
