import torch
import torch.nn as nn

class FusionOp(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
    def forward(self, 
               in0: torch.Tensor,  # 形状 [y_dim, 32, x_dim]
               in1: torch.Tensor,  # 形状 [x_dim]
               in2: torch.Tensor,  # 标量张量
               in3: torch.Tensor,  # 形状 [y_dim, 32]
               in4: torch.Tensor,  # 形状 [y_dim, 32, x_dim] 
               in5: torch.Tensor): # 形状 [32, x_dim]
        
        # 第一阶段计算
        tmp0 = in0.to(torch.float32)
        tmp1 = in1.expand_as(tmp0).to(torch.float32)
        tmp2 = tmp0 + tmp1
        
        tmp3 = in2.to(torch.float32)
        tmp4 = tmp3.view(1, 1)
        tmp5 = tmp2 / tmp4
        tmp6 = torch.sigmoid(tmp5)
        
        # 第二阶段计算
        tmp7 = in3.unsqueeze(2).expand(-1, -1, tmp0.shape[2]).to(torch.float32)
        tmp8 = tmp7
        
        # 符号计算
        tmp9 = (tmp8 > 0).to(torch.int8)
        tmp10 = (tmp8 < 0).to(torch.int8)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp11.to(tmp8.dtype)
        tmp13 = tmp12.to(torch.int32).to(torch.float32)
        
        # 混合计算
        tmp15 = tmp6 * tmp13
        tmp16 = in4.to(torch.float32)
        tmp17 = tmp15 * tmp16
        
        # 输出路径1
        output0 = tmp17.masked_fill_(tmp17.isnan(), 0).sum(dim=1)
        
        # 并行路径计算
        tmp22 = in5.expand_as(tmp16).to(torch.float32)
        tmp23 = tmp22 * tmp16
        output1 = tmp23.masked_fill_(tmp23.isnan(), 0).sum(dim=1)
        
        return output0.to(self.device), output1.to(self.device)

if __name__ == "__main__":
    y_dim = 4
    x_dim = 32
    r_dim = 32
    
    torch.manual_seed(2023)
    in0 = torch.randn(y_dim, r_dim, x_dim)  # 形状 [4,32,32]
    in1 = torch.randn(x_dim)               # 形状 [32]
    in2 = torch.tensor([0.5])              # 标量值
    in3 = torch.randn(y_dim, r_dim)        # 形状 [4,32]
    in4 = torch.randn(y_dim, r_dim, x_dim) # 形状 [4,32,32]
    in5 = torch.randn(r_dim, x_dim)        # 形状 [32,32]
    
    model = FusionOp(device='cpu')
    
    with torch.no_grad():
        out0, out1 = model(in0, in1, in2, in3, in4, in5)
    
    # 打印验证结果
    print("输出1形状:", out0.shape)  # 预期 [4,32]
    print("输出2形状:", out1.shape)  # 预期 [4,32]
    
    # 验证第一个样本的计算
    sample_idx = 0
    print("\n第一个样本输出1的前5个元素:", out0[sample_idx, :5])
    print("第一个样本输出2的前5个元素:", out1[sample_idx, :5])
