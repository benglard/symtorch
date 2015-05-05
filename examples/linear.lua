local symtorch = require '../symtorch'

local W = symtorch.Tensor(10,5):rand(0, 1)
local b = symtorch.Tensor(10,1)
local x1 = symtorch.Tensor(5,1):rand(0, 5)
local x2 = symtorch.Tensor(5,1):rand(0, 5)
local out = W:dot(x1) + b
print(out.w)
out.dw[1] = 1.0

print('Bias before update')
print(b.w)
local params = {W, b}
_graph:backward()
symtorch.update(params)
print('Bias after update')
print(b.w)

out = W:dot(x2) + b
print(out.w)
out.dw[2] = 1.0
print('Bias before update')
print(b.w)
_graph:backward()
symtorch.update(params)
print('Bias after update')
print(b.w)