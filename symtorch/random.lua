return { 
   rng = {
      binomial = function(size, p)
         local output = symtorch.Tensor(size)
         output.w:bernoulli(p)
         return output
      end,
   }
}