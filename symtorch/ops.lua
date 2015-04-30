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

local _scan = function(options)
   options = options or {}
   local fn = options.fn or function(...) end
   local seqs = options.sequences or {}
   local nsteps = options.nsteps or nil

   if nsteps == nil then
      nsteps = seqs[1].w:size(1)
      for i = 2, #seqs do
         local size = seqs[i].w:size(1)
         if nsteps > size then
            nsteps = size
         end
      end
   end

   local rv = {}
   for i = 1, nsteps do
      table.insert(rv, {fn(unpack(seqs))})
      seqs = rv[#rv]
   end
   return rv
end

local _update = Class { -- rmsprop by default
   __init__ = function(self)
      self.decayRate = 0.999
      self.epsilon = 1e-8
      self.stepCache = {}
   end,

   __call = function(self, params, lr, reg, clip)
      _graph:backward()

      lr = lr or 0.01
      reg = reg or 0.0001
      clip = clip or 5

      for i = 1, #params do
         local p = params[i]
         if self.stepCache[i] == nil then
            self.stepCache[i] = symtorch.Tensor(p.w:size())
         end
         local s = self.stepCache[i]

         -- clip gradients
         p.dw:apply(function(elem)
            if elem > clip then return clip
            elseif elem < -clip then return -clip
            else return elem end
         end)

         -- update cache
         s.w:mul(self.decayRate)
            :add(torch.cmul(p.dw, p.dw):mul(1 - self.decayRate))

         -- update params
         local delta = torch.mul(p.dw, -lr)
                        :cdiv(torch.add(s.w, self.epsilon):sqrt())
                        :add(torch.mul(p.w, -reg))
         p.w:add(delta)
         p.dw:zero()
      end
   end,
}

return {
   tanh = _tanh,
   relu = _relu,
   sigmoid = _sigmoid,
   exp = _exp,
   softmax = _softmax,
   scan = _scan,
   update = _update()
}