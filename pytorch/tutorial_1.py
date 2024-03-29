import torch
import numpy as np

# tensor initiate, data to tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# numpy to tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor
x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand} \n")

# initiate with random, constant value
shape = (2,3) # dimension
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# attribute of tensor
tensor = torch.rand(3,4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# operation
# indexing, slicing
tensor = torch.ones(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# concatenate
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)
t2 = torch.stack([tensor, tensor, tensor], dim=0)
# print(t2)

# matrix multiplication. y1, y2, y3 all same res
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# element-wise product. z1, z2, z3 all same res
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum() # maybe other options...
agg_item = agg.item()
print(agg_item, type(agg_item))

# bridge - conv to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# tensor, numpy share memory
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)
# tensor, numpy share memory
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")