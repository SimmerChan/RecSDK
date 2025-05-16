import torch
import numpy as np

class SliceSumCat(torch.nn.Module):
    def __init__(self, indices):
        """
        indices: 切片索引张量，形状为 [output_num, 2]
                 每行包含 [start_idx, end_idx]
        """
        super().__init__()
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
    
    def forward(self, x):
        """
        x: 输入张量，形状为 [batch_size, row, col]
        输出形状为 [batch_size, output_num, col]
        """
        outputs = []
        # 遍历每个切片
        for i in range(len(self.indices)):
            start = self.indices[i, 0]
            end = self.indices[i, 1]
            # 提取切片并求和
            sliced = x[:, start:end, :]  # [B, S, C]
            summed = sliced.sum(dim=1)   # [B, C]
            outputs.append(summed)
        # 沿新维度拼接结果
        return torch.stack(outputs, dim=1)  # [B, N, C]

if __name__ == "__main__":
    batch_size = 2
    row = 10
    col = 5
    output_num = 3
    indices = np.array([[1,3], [4,7], [8,9]])  # 3个切片

    torch.manual_seed(0)
    input_tensor = torch.randint(0,10,(batch_size, row, col)).float()
    
    model = SliceSumCat(indices)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output.shape)
    print("\n第一个样本的验证:")
    print("原始数据:\n", input_tensor[0])
    print("\n切片1(1:3)求和:\n", input_tensor[0,1:3].sum(dim=0))
    print("切片2(4:7)求和:\n", input_tensor[0,4:7].sum(dim=0))
    print("切片3(8:9)求和:\n", input_tensor[0,8:9].sum(dim=0))
    print("\n模型输出:\n", output[0])
