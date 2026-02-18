import torch

def Loss_Calculation(outputs, batch):
    return MSE_Loss(outputs, batch)

def MSE_Loss(outputs, batch):
    return torch.mean((outputs - batch)**2, dim=(1, 2)) 

def Power_MSE_Loss(outputs, batch):
    return torch.mean((outputs - batch)**4, dim=(1, 2)) 

def MAE_Loss(outputs, batch):
    return torch.mean(torch.abs(outputs - batch), dim=(1, 2))

def RMSE_Loss(outputs, batch):
    return torch.sqrt(torch.mean((outputs - batch)**2, dim=(1, 2)) + 1e-6)

def Huber_Loss_Score(outputs, batch, delta=1.0):
    errors = torch.abs(outputs - batch)
    # Huber 로직: 오차가 delta보다 작으면 0.5 * error^2, 크면 delta * (error - 0.5 * delta)
    quad = torch.min(errors, torch.tensor(delta).to(errors.device))
    lin = errors - quad
    
    # 샘플별 평균 계산 (batch_size 크기의 벡터 반환)
    sample_huber = (0.5 * quad**2 + delta * lin).mean(dim=(1, 2))
    return sample_huber
