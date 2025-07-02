import torch

num_dim_x = 12
num_dim_control = 4

g = 9.81
m_p = 0.5  # mass of the payload
m_q = 1.63  # mass of the quadrotor

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]
    r_tilde_x, r_tilde_y, r_q_x, r_q_y, r_q_z, l, v_tilde_x, v_tilde_y, v_q_x, v_q_y, v_q_z, l_dot = [x[:,i,0] for i in range(num_dim_x)]
    
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    vel = torch.stack([v_tilde_x, v_tilde_y, v_q_x, v_q_y, v_q_z, l_dot], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    f[:, 0:6, 0] = vel.squeeze(-1)
    
    # Define v_tilde vector
    v_tilde = torch.stack([v_tilde_x, v_tilde_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    
    # Define normal vector of cable
    r_tilde = torch.stack([r_tilde_x, r_tilde_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    n = torch.cat([r_tilde, -torch.sqrt(1-torch.norm(r_tilde, dim=1)).unsqueeze(-1)], dim=1) # (bs, 3, 1)

    
    # Define B matrix for slung payload
    I2 = torch.eye(2, device=x.device).repeat(bs, 1, 1) # shape (bs, 2, 2)
    r_tilde_row = r_tilde.view(bs, 1, 2)
    last_row = r_tilde_row/torch.sqrt(1-torch.norm(r_tilde))
    B = torch.cat([I2, last_row], dim=1)
    
    # Define B_dot matrix for slung payload
    O2 = torch.zeros(bs, 2, 2).type(x.type())  # shape (bs, 2, 2)
    B_dot31 = 1/torch.sqrt(1-torch.norm(r_tilde)) + (r_tilde_x ** 2 + r_tilde_x * r_tilde_y)/(1-torch.norm(r_tilde)) ** (3/2)
    B_dot31 = B_dot31 * v_tilde_x
    B_dot32 = 1/torch.sqrt(1-torch.norm(r_tilde)) + (r_tilde_y ** 2 + r_tilde_x * r_tilde_y)/(1-torch.norm(r_tilde)) ** (3/2)
    B_dot32 = B_dot32 * v_tilde_y
    B_dot3 = torch.stack([B_dot31, B_dot32], dim=1).unsqueeze(1) # (bs, 1, 2)
    B_dot = torch.cat([O2, B_dot3], dim=1)  # shape (bs, 3, 2)
    
    # Kane's method   
    # Define M matrix for slung payload
    M = torch.zeros(bs, 6, 6).type(x.type())
    M11 = m_p * (l ** 2).view(-1, 1, 1) * torch.bmm(B.transpose(1, 2), B)
    M12 = m_p * l.view(-1, 1, 1) * B.transpose(1, 2)
    M13 = torch.zeros(bs, 2, 1).type(x.type())
    M21 = m_p * l.view(-1, 1, 1) * B
    M22 = (m_p + m_q) * torch.eye(3, device=x.device).repeat(bs, 1, 1)
    M23 = m_p * n
    M31 = torch.zeros(bs, 1, 2).type(x.type())
    M32 = m_p * n.transpose(1, 2)
    M33 = torch.tensor([1], device=x.device).repeat(bs, 1, 1)  # (bs, 1, 1)
    
    # Concactenate M matrix: (bs, 6, 6)
    M = torch.cat([
        torch.cat([M11, M12, M13], dim=2),
        torch.cat([M21, M22, M23], dim=2),
        torch.cat([M31, M32, M33], dim=2)
    ], dim=1)
    
    # Compute the inverse of M
    M_inv = torch.inverse(M)
    
    # Compute C matrix
    C = torch.zeros(bs, 6, 6).type(x.type())
    C11 = (l ** 2).view(-1, 1, 1) * torch.bmm(B_dot.transpose(1, 2), B_dot) + l_dot.view(-1, 1, 1) * l.view(-1, 1, 1) * torch.bmm(B.transpose(1, 2), B)
    C12 = torch.zeros(bs, 2, 3).type(x.type())
    C13 = l.view(-1, 1, 1) * torch.bmm(torch.bmm(B.transpose(1, 2), B), v_tilde)
    C21 = l.view(-1, 1, 1) * B_dot + B * l_dot.view(-1, 1, 1)
    C22 = torch.zeros(bs, 3, 3).type(x.type())
    C23 = torch.bmm(B, v_tilde)
    C31 = l.view(-1, 1, 1) * torch.bmm(n.transpose(1, 2), B_dot)
    C32 = torch.zeros(bs, 1, 3).type(x.type())
    C33 = torch.zeros(bs, 1, 1).type(x.type())
    
    C = torch.cat([
        torch.cat([C11, C12, C13], dim=2),
        torch.cat([C21, C22, C23], dim=2),
        torch.cat([C31, C32, C33], dim=2)
    ], dim=1)
    
    C = m_p * C
    
    # Compute G matrix
    G = torch.zeros(bs, 6, 3).type(x.type())
    G1 = m_p * l.view(-1, 1, 1) * B.transpose(1, 2)
    G2 = (m_p + m_q) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    G3 = m_p * n.transpose(1, 2)
    G = torch.cat([G1, G2, G3], dim=1)
    g_I = torch.tensor([0, 0, -g], device=x.device).view(1, 3, 1).expand(bs, 3, 1)  # (bs, 3, 1)
    F_g = torch.bmm(G, g_I)
    
    f[:, 6:12, 0] = torch.bmm(M_inv, (F_g - torch.bmm(C, vel))).squeeze(-1)

    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    bs = x.shape[0]
    
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())
    B[:, 8, 0] = 1  # Thrust force in x direction of earth frame
    B[:, 9, 1] = 1  # Thrust force in y direction of earth frame
    B[:, 10, 2] = 1  # Thrust force in z direction of earth frame
    B[:, 11, 3] = -1  # Retracting torque

    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')
