local symtorch = require '../symtorch'

local W = symtorch.Tensor(10,5):rand(0, 1)
local b = symtorch.Tensor(10,1)
local x = symtorch.Tensor(5,1):rand(0, 5)
local out = W:dot(x) + b
print(out.w)
out.dw[1] = 1.0

local params = {W, b}
symtorch.update(params)