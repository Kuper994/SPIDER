import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(DEVICE)
N_EPOCHS = 15
LEARNING_RATE = 1e-3
torch.manual_seed(seed=1234)
