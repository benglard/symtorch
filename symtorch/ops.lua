local ffi = require 'ffi'
local libsymtorch = ffi.load(symtorch.cpath)
ffi.cdef[[ 
void tensor_sigmoid(double* tensor, int size);
void tensor_sigmoid_backward(
  double* input_dw,
  double* output_w,
  double* output_dw,
  unsigned int size
);

void tensor_relu(double* tensor, int size);
void tensor_relu_backward(
  double* input_w,
  double* input_dw,
  double* output_dw,
  unsigned int size
);
]]

local _tanh = function(input)
   local output = input:clone()
   output.w:tanh()

   _graph:add(function()
      local ow2 = torch.cmul(output.w, output.w)
      local delta = torch.ones(ow2:size()) - ow2
      delta:cmul(output.dw)
      input.dw:add(delta)
   end)

   return output
end

local _relu = function(input)
   local output = input:clone()
   libsymtorch.tensor_relu(output.w:data(), output.w:nElement())
   
   _graph:add(function()
      libsymtorch.tensor_relu_backward(
         input.w:data(),
         input.dw:data(),
         output.dw:data(),
         output.w:nElement()
      )
   end)

   return output
end

local _sigmoid = function(input)
   local output = input:clone()
   libsymtorch.tensor_sigmoid(output.w:data(), output.w:nElement())
   
   _graph:add(function()
      libsymtorch.tensor_sigmoid_backward(
         input.dw:data(),
         output.w:data(),
         output.dw:data(),
         output.w:nElement()
      )
   end)

   return output
end

local _exp = function(input)
   local output = input:clone()
   output.w:exp()

   _graph:add(function()
      input.dw:cmul(output.dw)
   end)

   return output
end

local _softmax = function(input)
   local output = input:clone()
   local max = output.w:max()
   output.w:add(-max):exp():div(output.w:sum())
   return output
end

return {
   tanh = _tanh,
   relu = _relu,
   sigmoid = _sigmoid,
   exp = _exp,
   softmax = _softmax
}