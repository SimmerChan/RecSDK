import torch
import torch.nn as nn

class ComplexFusionOp(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
    def forward(self, 
                in0: torch.Tensor,  # 形状 [y_dim, 32, x_dim]
                in1: torch.Tensor,  # 形状 [x_dim]
                in2: torch.Tensor,  # 标量张量
                in3: torch.Tensor,  # 形状 [y_dim, 64]
                in4: torch.Tensor,  # 形状 [y_dim, 32, x_dim]
                in5: torch.Tensor): # 形状 [32, x_dim]
        
        # 主计算路径
        tmp0 = in0.to(torch.float32)
        tmp1 = in1.expand_as(tmp0[..., 0]).unsqueeze(1).expand_as(tmp0)
        tmp2 = tmp0 + tmp1
        
        tmp3 = in2.to(torch.float32)
        tmp5 = tmp2 / tmp3
        tmp6 = torch.sigmoid(tmp5)
        
        # 符号计算分支
        tmp7 = in3.unsqueeze(2).expand(-1, -1, tmp0.shape[2]).to(torch.float32)
        sign = torch.sign(tmp7)
        
        # 混合计算
        tmp15 = tmp6 * sign
        tmp16 = in4.to(torch.float32)
        tmp17 = tmp15 * tmp16
        output0 = tmp17.sum(dim=1)
        
        # 并行计算路径
        tmp22 = in5.expand_as(tmp16).to(torch.float32)
        tmp23 = tmp22 * tmp16
        output1 = tmp23.sum(dim=1)
        
        return output0.to(self.device), output1.to(self.device)

if __name__ == "__main__":
    # 配置参数
    y_dim = 4
    x_dim = 32
    r_dim = 64
    
    # 生成测试数据
    torch.manual_seed(2023)
    in0 = torch.randn(y_dim, 32, x_dim) * 2  # 形状 [4,32,32]
    in1 = torch.randn(x_dim)                 # 形状 [32]
    in2 = torch.tensor([0.5])                # 标量参数
    in3 = torch.randn(y_dim, r_dim)          # 形状 [4,64]
    in4 = torch.randn(y_dim, 32, x_dim)      # 形状 [4,32,32]
    in5 = torch.randn(32, x_dim)             # 形状 [32,32]
    
    model = ComplexFusionOp(device='cpu')
    
    with torch.no_grad():
        out0, out1 = model(in0, in1, in2, in3, in4, in5)
    
    # 结果验证
    print("输出1形状:", out0.shape)  # 预期 [4,32]
    print("输出2形状:", out1.shape)  # 预期 [4,32]
    
    # 计算过程验证
    sample_idx = 0
    manual_out0 = (torch.sigmoid((in0[sample_idx] + in1) / 0.5) 
                   * torch.sign(in3[sample_idx].unsqueeze(1)[:, :32]) 
                   * in4[sample_idx]).sum(dim=0)
    
    print("\n输出1手工计算:", manual_out0[:5])
    print("模型计算结果:", out0[sample_idx, :5])
    print("最大差异:", torch.max(torch.abs(manual_out0 - out0[sample_idx])))
