import torch
import torch.nn as nn

class SliceLowKernel(nn.Module):
    def __init__(self, input_shape, start_indices, slice_len):
        """
        input_shape: 输入张量形状 (row, col)
        start_indices: 切片起始索引列表
        slice_len: 每个切片长度
        """
        super().__init__()
        self.register_buffer('start_indices', torch.tensor(start_indices, dtype=torch.long))
        self.slice_len = slice_len
        self.output_shape = (len(start_indices), input_shape[0], slice_len)
        
    def forward(self, x):
        """
        x: 输入张量形状 [batch, row, col]
        输出形状 [batch, num_slices, row, slice_len]
        """
        batch_size = x.size(0)
        outputs = []
        
        for idx in self.start_indices:
            # 计算列范围
            start_col = idx.item()
            end_col = start_col + self.slice_len
            
            # 提取切片
            sliced = x[..., start_col:end_col]  # [B, R, L]
            
            # 维度处理
            outputs.append(sliced.unsqueeze(1))  # 添加切片维度
            
        # 拼接所有切片
        return torch.cat(outputs, dim=1)  # [B, S, R, L]

if __name__ == "__main__":
    batch_size = 2
    input_row = 1000
    input_col = 2048
    slice_num = 4
    slice_len = 128
    start_indices = [256, 512, 768, 1024]  # 4个切片起始位置
    
    torch.manual_seed(2023)
    input_tensor = torch.randn(batch_size, input_row, input_col)
    
    model = SliceLowKernel(
        input_shape=(input_row, input_col),
        start_indices=start_indices,
        slice_len=slice_len
    )
    
    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 结果验证
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)
    
    # 验证第一个切片的正确性
    test_slice_idx = 0
    manual_slice = input_tensor[0, :, start_indices[0]:start_indices[0]+slice_len]
    model_slice = output[0, test_slice_idx]
    
    print("\n手工切片形状:", manual_slice.shape)
    print("模型切片形状:", model_slice.shape)
    print("数据一致性检查:", torch.allclose(manual_slice, model_slice, atol=1e-7))
