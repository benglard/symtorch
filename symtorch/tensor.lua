return {
   Tensor = Class {
      name = 'Tensor',

      __init__ = function(self, ...)
         self.w = torch.zeros(...)
         self.dw = torch.zeros(...)
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
         local rv = {}
         setmetatable(rv, getmetatable(self))
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
         output.w:cmul(self.w, other.w)
         output.dw:resizeAs(output.w)
      
         _graph:add(function()
            self.dw:add(torch.cmul(other.w, output.dw))
            other.dw:add(torch.cmul(self.w, output.dw))
         end)

         return output
      end,

      dot = function(self, other) -- matrix multiply
         local output = self:clone()
         output.w = self.w * other.w
         output.dw:resizeAs(output.w)

         _graph:add(function()
            self.dw:add(other.w * output.dw:t())
            other.dw:add(self.w:t() * output.dw)
         end)

         return output
      end
   }
}