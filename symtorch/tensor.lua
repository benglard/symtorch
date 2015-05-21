return {
   Tensor = Class {
      name = 'Tensor',

      __init__ = function(self, ...)
         self.w = torch.DoubleTensor(...):fill(0)
         self.dw = torch.DoubleTensor(...):fill(0)
      end,

      __index = function(self, key)
         return self.w[key]
      end,

      rand = function(self, mu, std)
         mu = mu or 0
         std = std or math.sqrt(1.0/self.w:nElement())
         self.w:apply(function() return torch.normal(mu, std) end)
         return self
      end,

      copy = function(self, other)
         if other:isTensor() then
            self.w:copy(other)
         else
            self.w:copy(other.w)
            self.dw:copy(other.dw)
         end
         return self
      end,

      clone = function(self)
         local rv = setmetatable({}, getmetatable(self))
         for key, val in pairs(self) do
            if torch.type(val) == 'torch.DoubleTensor' then
               rv[key] = val:clone()
            else
               rv[key] = val
            end
         end
         return rv
      end,

      flatten = function(self)
         self.w:resize(self.w:nElement())
         self.dw:resize(self.dw:nElement())
         return self
      end,

      fill = function(self, val)
         self.w:fill(val)
         return self
      end,

      resize = function(self, shape)
         self.w:resize(shape)
         self.dw:resize(shape)
      end,

      resizeAs = function(self, other)
         self.w:resizeAs(other.w)
         self.dw:resizeAs(other.dw)
      end,

      size = function(self, ...)
         return self.w:size(...)
      end,

      __add = function(self, other) -- element wise
         local output = symtorch.Tensor()
         output.w:add(self.w, other.w)
         output.dw:resizeAs(output.w):zero()

         _graph:add(function()
            self.dw:add(output.dw)
            other.dw:add(output.dw)
         end)

         return output
      end,

      __mul = function(self, other) -- element wise
         local output = symtorch.Tensor()
         output.w:cmul(self.w, other.w)
         output.dw:resizeAs(output.w):zero()
      
         _graph:add(function()
            self.dw:addcmul(other.w, output.dw)
            other.dw:addcmul(self.w, output.dw)
         end)

         return output
      end,

      dot = function(self, other) -- matrix multiply
         local output = symtorch.Tensor()
         output.w = self.w * other.w
         output.dw:resizeAs(output.w):zero()

         _graph:add(function()
            local nDim = output.w:dim()
            if nDim == 1 then
               local delta = torch.Tensor(output.dw:size(1), other.w:size(1))
               delta:addr(0, 1, output.dw, other.w)
               self.dw:add(delta)
               other.dw:addmv(self.dw:t(), output.w)
            elseif nDim == 2 then
               self.dw:addmm(output.dw, other.w:t())
               other.dw:addmm(self.w:t(), output.dw)
            elseif nDim == 3 then
               self.dw:addbmm(other.w, output.dw:transpose(2, 3))
               other.dw:addbmm(self.w:transpose(2, 3), output.dw)
            end
         end)

         return output
      end,

      depthAdd = function(self, other) -- add along depth
         local safe = symtorch.Tensor(self:size())
         local depth = other:size(1)
         for d = 1, depth do
            safe[d]:fill(other[d])
         end

         _graph:add(function()
            for d = 1, depth do
               local grad = safe.dw[d]:sum()
               other.dw[d] = other.dw[d] + grad
            end
         end)

         return self + safe
      end,

      index = function(self, idx)
         -- Pulls out a row as column vector
         -- Useful for building lookup tables

         local output = symtorch.Tensor()
         output.w:copy(self[idx])
         output.dw:resizeAs(output.w):zero()

         _graph:add(function()
            self.dw:add(output.dw[idx])
         end)

         return output
      end,
   }
}