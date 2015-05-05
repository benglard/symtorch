return {
   Tensor = Class {
      name = 'Tensor',

      __init__ = function(self, ...)
         self.w = torch.Tensor(...):fill(0)
         self.dw = torch.Tensor(...):fill(0)
      end,

      __index = function(self, key)
         return self.w[key]
      end,

      rand = function(self, mu, std)
         mu = mu or 0
         std = std or 1.0/math.sqrt(self.w:nElement())
         self.w:apply(function() return torch.normal(mu, std) end)
         return self
      end,

      copy = function(self, other)
         other.w:resizeAs(self.w)
         other.dw:resizeAs(self.dw)
         self.w:copy(other.w)
         self.dw:copy(other.dw)
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

      copy = function(self, w, dw)
         if w ~= nil then self.w:resizeAs(w):copy(w) end
         if dw ~= nil then self.dw:resizeAs(dw):copy(dw) end
         return self
      end,

      __add = function(self, other) -- element wise
         local output = self:clone()         
         output.w:add(other.w)
         output.dw:resizeAs(output.w)

         _graph:add(function()
            self.dw:add(output.dw)
            other.dw:add(output.dw)
         end)

         return output
      end,

      __mul = function(self, other) -- element wise
         local output = self:clone()
         output.w:cmul(other.w)
         output.dw:resizeAs(output.w)
      
         _graph:add(function()
            self.dw:addcmul(other.w, output.dw)
            other.dw:addcmul(self.w, output.dw)
         end)

         return output
      end,

      dot = function(self, other) -- matrix multiply
         local output = self:clone()
         output.w = self.w * other.w
         output.dw:resizeAs(output.w)

         _graph:add(function()
            local nDim = output.w:dim()
            if nDim == 1 then
               self.dw:addr(output.dw, other.w)
               other.dw:addmv(self.dw:t(), output.w)
            elseif nDim == 2 then
               self.dw:addmm(output.dw, other.w:t())
               other.dw:addmm(self.w:t(), output.dw)
            elseif nDim == 3 then
               self.dw:addbmm(other.w, output.dw:transpose(2, 3))
               other.dw:addbmm(self.w:transpose(2, 3) * output.dw)
            end
         end)

         return output
      end
   }
}