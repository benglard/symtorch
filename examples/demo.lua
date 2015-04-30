local symtorch = require '../symtorch'
print(symtorch)

local t = symtorch.Tensor(10, 1, 28, 28) -- MNIST, Batch: 10
print(t)
local t2 = symtorch.Tensor(5, 5)
print(t2.w)
t2[1][1] = 5
print(t2.w)
print('-----\n')

local randt = symtorch.Tensor(5, 5):rand(0, 0.8)
print(randt)
print(randt.w)
print(randt.dw)
print('-----\n')

--local graph = symtorch.Graph()
local input = symtorch.Tensor(5, 5):rand(0, 0.8)
local output = symtorch.tanh(input)
print(output.w)
local output = symtorch.relu(output)
local output = symtorch.sigmoid(output)
print(output.w)
local output = output + randt
print(output.w)
local output = output * randt
print(output.w)
local output = output:dot(randt)
print(output.w)
local output = symtorch.softmax(output)
print(output.w, output.w:sum())