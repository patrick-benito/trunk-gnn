import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

velocity_mask = torch.vstack((torch.eye(3)*0, 
                              torch.eye(3)*1))
position_mask = torch.vstack((torch.eye(3)*1,
                              torch.eye(3)*0))

velocity_mask = velocity_mask.to(device)
position_mask = position_mask.to(device)