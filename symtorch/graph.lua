return {
   Graph = Class {
      backprop = {},

      __init__ = function(self, nb)
         self.needsBackprop = nb or true
      end,

      add = function(self, cb)
         if self.needsBackprop then
            table.insert(self.backprop, cb) 
         end
      end,

      backward = function(self)
         for i = #self.backprop, 1, -1 do
            self.backprop[i]()
         end
         self.backprop = {}
      end
   }
}