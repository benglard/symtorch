local symtorch = require '../symtorch'

local x = symtorch.Tensor(1, 24, 24):rand(0, 1)
local o

local function fill_across(src, dest)
   --[[
      Ideally we would want bias to be a vector
      but since torch.add only allows additions
      of the same size tensors, conv1_b for instance,
      has to be of size 8,24,24. So we can create a 
      src tensor of size 8 and set each 24,24 subtensor
      to src[i]
   ]]
   local size = src:size(1)
   for i = 1, size do
      dest[i]:fill(src[i])
   end
end

local conv1_W = symtorch.Tensor(8, 5, 5):rand()
local conv1_b_real = symtorch.Tensor(8)
local conv1_b = symtorch.Tensor(8, 24, 24)
fill_across(conv1_b_real.w, conv1_b.w)

o = symtorch.conv2d(x, conv1_W, 1, 2) + conv1_b
o = symtorch.relu(o)
o = symtorch.maxpool2d(o, 2, 2, 2)

local conv2_W = symtorch.Tensor(16, 5, 5):rand()
local conv2_b_real = symtorch.Tensor(16)
local conv2_b = symtorch.Tensor(16, 12, 12)
fill_across(conv2_b_real.w, conv2_b.w)

o = symtorch.conv2d(o, conv2_W, 1, 2) + conv2_b
o = symtorch.relu(o)
o = symtorch.maxpool2d(o, 3, 3, 3)

local size = 16*4*4
o.w:resize(size, 1)
o.dw:resize(size, 1)

local fc_W = symtorch.Tensor(10, size):rand()
local fc_b = symtorch.Tensor(10)
o = fc_W:dot(o) + fc_b
local sm = symtorch.softmax(o)
print(sm.w) -- should all be around 0.1 (all equiprobable)
local val, argmax = sm.w:max(1)
argmax = argmax:squeeze()
print('Argmax: ', argmax, 'Max:', val:squeeze(), '\n\n')
o.dw[argmax] = 1.0

print('Bias before update')
print(fc_b.w)
local params = {conv1_W, conv1_b_real, conv2_W, conv2_b_real, fc_W, fc_b}
symtorch.update(params)
print('Bias after update')
print(fc_b.w)