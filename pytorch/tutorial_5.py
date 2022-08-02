import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

# also can use "x.requires_grad_(True)" method later
w = torch.randn(5, 3, requires_grad=True) # parameters
b = torch.randn(3, requires_grad=True) # parameters

z = torch.matmul(x, w)+b # actual computation
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# calculate gradient - weight optimization
loss.backward()
print(w.grad)
print(b.grad)

# stop backpropagation
# Common scenario when fine-tuning pre-trained neural networks.
# Computation speed is improved when only the propagation phase is performed.
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# use "detach" method
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)