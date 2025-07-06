import torch

num_dim_x = 12
num_dim_control = 4

g = 9.81
m_p = 0.5  # mass of the payload
m_q = 1.63  # mass of the quadrotor

# Add another function to get dynamics simplify f_func, B_func

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]
    # r_p_x, r_p_y, r_q_x, r_q_y, r_q_z, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]
    r_q_x, r_q_y, r_q_z, r_p_x, r_p_y, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]  # For state reversal
    
    # Define r_p, v_p vector
    r_p = torch.stack([r_p_x, r_p_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_p = torch.stack([v_p_x, v_p_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_p_T = r_p.transpose(1, 2)  # (bs, 1, 2)
    v_p_T = v_p.transpose(1, 2)  # (bs, 1, 2)
    
    # Define v_q vector
    v_q = torch.stack([v_q_x, v_q_y, v_q_z], dim=1).unsqueeze(-1) # (bs, 3, 1)
    
    # Define l, l_dot
    l = l.view(-1, 1, 1)  # (bs, 1, 1)
    l_dot = l_dot.view(-1, 1, 1)  # (bs, 1, 1)
    
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    vel = torch.cat([v_p, v_q, l_dot], dim=1)  # (bs, 3, 1)
    vel_r = torch.cat([v_q, v_p, l_dot], dim=1)  # for state reversal
    f[:, 0:6, 0] = vel_r.squeeze(-1)
    
    # Define normal vector of cable
    Z = -torch.sqrt(1-torch.bmm(r_p_T, r_p))  # ENU frame
    n = torch.cat([r_p, Z], dim=1)  # (bs, 3, 1)
    n_T = n.transpose(1, 2)
    
    # Define B matrix for slung payload
    I2 = torch.eye(2).repeat(bs, 1, 1).type(x.type())
    B = torch.cat([I2, -r_p_T/Z], dim=1)  # (bs, 3, 2)
    B_T = B.transpose(1, 2)
    
    # Define B_dot matrix for slung payload
    O2 = torch.zeros(bs, 2, 2).type(x.type())
    B_dot = torch.cat([O2, -(Z ** 2 * v_p_T + torch.bmm(r_p_T, v_p) * r_p_T)/(Z ** 3)], dim=1)  # shape (bs, 3, 2)
    
    # Kane's method
    # Define M matrix for slung payload
    M = torch.zeros(bs, 6, 6).type(x.type())
    M11 = m_p * (l ** 2) * torch.bmm(B_T, B)
    M12 = m_p * l * B_T
    M13 = torch.zeros(bs, 2, 1).type(x.type())
    M22 = (m_p + m_q) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M23 = m_p * n
    M33 = m_p * torch.tensor([1]).repeat(bs, 1, 1).type(x.type())  # (bs, 1, 1)
     
    # Concactenate M matrix: (bs, 6, 6)
    M = torch.cat([
        torch.cat([M11, M12, M13], dim=2),
        torch.cat([M12.transpose(1, 2), M22, M23], dim=2),
        torch.cat([M13.transpose(1, 2), M23.transpose(1, 2), M33], dim=2)
    ], dim=1)
   
    # Compute C matrix
    C = torch.zeros(bs, 6, 6).type(x.type())
    C11 = (l ** 2) * torch.bmm(B_T, B_dot) + l_dot * l * torch.bmm(B_T, B)
    C12 = torch.zeros(bs, 2, 3).type(x.type())
    C13 = l * torch.bmm(torch.bmm(B_T, B), v_p)
    C21 = l * B_dot + B * l_dot
    C22 = torch.zeros(bs, 3, 3).type(x.type())
    C23 = torch.bmm(B, v_p)
    C31 = l * torch.bmm(n_T, B_dot)
    C32 = torch.zeros(bs, 1, 3).type(x.type())
    C33 = torch.zeros(bs, 1, 1).type(x.type())
    
    C = m_p * torch.cat([
        torch.cat([C11, C12, C13], dim=2),
        torch.cat([C21, C22, C23], dim=2),
        torch.cat([C31, C32, C33], dim=2)
    ], dim=1)
    
    # Compute G matrix
    G = torch.zeros(bs, 6, 3).type(x.type())
    G1 = m_p * l.view(-1, 1, 1) * B_T
    # G2 = (m_p + m_q) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    # G3 = m_p * n.transpose(1, 2)
    # Training for delta f and delta tau
    G2 = torch.zeros(bs, 3, 3).type(x.type())
    G3 = torch.zeros(bs, 1, 3).type(x.type())
    G = torch.cat([G1, G2, G3], dim=1)
    g_I = torch.tensor([0, 0, -g]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())  # (bs, 3, 1)
    F_g = torch.bmm(G, g_I)
    
    f[:, 6:12, 0] = torch.linalg.solve(M, F_g - torch.bmm(C, vel)).squeeze(-1)
    
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    bs = x.shape[0]
    # r_p_x, r_p_y, r_q_x, r_q_y, r_q_z, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]
    r_q_x, r_q_y, r_q_z, r_p_x, r_p_y, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]  # For state reversal
    
    # Define r_p, v_p vector
    r_p = torch.stack([r_p_x, r_p_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_p_T = r_p.transpose(1, 2)  # (bs, 1, 2)
    
    # Define l
    l = l.view(-1, 1, 1)  # (bs, 1, 1)
    
    # Define normal vector of cable
    Z = -torch.sqrt(1-torch.bmm(r_p_T, r_p))  # ENU frame
    n = torch.cat([r_p, Z], dim=1)  # (bs, 3, 1)
    
    # Define B matrix for slung payload
    I2 = torch.eye(2).repeat(bs, 1, 1).type(x.type())
    B = torch.cat([I2, -r_p_T/Z], dim=1)  # (bs, 3, 2)
    B_T = B.transpose(1, 2)
    
    # Kane's method
    # Define M matrix for slung payload
    M = torch.zeros(bs, 6, 6).type(x.type())
    M11 = m_p * (l ** 2) * torch.bmm(B_T, B)
    M12 = m_p * l * B_T
    M13 = torch.zeros(bs, 2, 1).type(x.type())
    M22 = (m_p + m_q) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M23 = m_p * n
    M33 = m_p * torch.tensor([1]).repeat(bs, 1, 1).type(x.type())  # (bs, 1, 1)
     
    # Concactenate M matrix: (bs, 6, 6)
    M = torch.cat([
        torch.cat([M11, M12, M13], dim=2),
        torch.cat([M12.transpose(1, 2), M22, M23], dim=2),
        torch.cat([M13.transpose(1, 2), M23.transpose(1, 2), M33], dim=2)
    ], dim=1)
    

    H = torch.zeros(bs, 6, num_dim_control).type(x.type())
    H[:, 2, 0] = 1  # Thrust force in x direction of earth frame
    H[:, 3, 1] = 1  # Thrust force in y direction of earth frame
    H[:, 4, 2] = 1  # Thrust force in z direction of earth frame
    H[:, 5, 3] = 1  # Extending torque

    B = torch.cat([torch.zeros(bs, 6, num_dim_control).type(x.type()), torch.linalg.solve(M, H)], dim=1)
    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')


# How to calculate Bbot: ????
def Bbot_func(B):
    # Compute Bbot
    Bbot = []
    for i in range(B.shape[0]):
        gi = B[i]  # num_dim_x x num_dim_control
        # SVD: Bi = U S Vh
        U, S, Vh = torch.linalg.svd(gi, full_matrices=True)
        # Null space: columns of U corresponding to zero singular values
        # For numerical stability, use a tolerance
        tol = 1e-7
        null_mask = S < tol
        if null_mask.sum() == 0:
            # If no exact zeros, take the last (n-m) columns of U
            Bbot_i = U[:, num_dim_control:]
        else:
            Bbot_i = U[:, null_mask]
        Bbot.append(Bbot_i)
    # Stack to shape bs x n x (n-m)
    Bbot = torch.stack(Bbot, dim=0).type(B.type())
    return Bbot
