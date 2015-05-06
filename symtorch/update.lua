local _update = Class { -- rmsprop by default
   decayRate = 0.999,
   epsilon = 1e-8,
   stepCache = {},
   lr = 0.01,
   reg = 0.0001,
   clip = 5,

   __call = function(self, params)
      for i = 1, #params do
         local p = params[i]
         if self.stepCache[i] == nil then
            self.stepCache[i] = symtorch.Tensor(p.w:size())
         end
         local s = self.stepCache[i]

         -- clip gradients
         p.dw:apply(function(elem)
            if elem > self.clip then return self.clip
            elseif elem < -self.clip then return -self.clip
            else return elem end
         end)

         -- update cache
         s.w:mul(self.decayRate)
            :add(torch.cmul(p.dw, p.dw):mul(1 - self.decayRate))

         -- update params
         local delta = torch.mul(p.dw, -self.lr)
                        :cdiv(torch.add(s.w, self.epsilon):sqrt())
                        :add(torch.mul(p.w, -self.reg))
         p.w:add(delta)
         p.dw:zero()
      end
   end,
}

return { update = _update() }