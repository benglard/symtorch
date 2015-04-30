local symtorch = require '../symtorch'

local Linear = Class {
   __init__ = function(self, input_size, output_size)
      self.W = symtorch.Tensor(output_size, input_size):rand(0, 1)
      self.b = symtorch.Tensor(output_size)
   end,

   forward = function(self, input)
      return self.W:dot(input) + self.b
   end
}

local Reshape = Class {
   __init__ = function(self, shape)
      self.shape = shape
   end,

   forward = function(self, input)
      input.w:reshape(self.shape)
      return input
   end
}

local Softmax = Class {
   forward = function(self, input)
      return symtorch.softmax(input)
   end
}

local layer1 = Reshape(4)
local layer2 = Linear(4, 10)
local layer3 = Softmax()

local x = symtorch.Tensor(4,1):rand(0,1)
local ol1 = layer1:forward(x)
local ol2 = layer2:forward(ol1)
local ol3 = layer3:forward(ol2)
print(ol3.w)