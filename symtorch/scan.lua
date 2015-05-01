local _scan = function(options)
   options = options or {}
   local fn = options.fn or function(...) end
   local seqs = options.sequences or {}
   local outputs = options.outputs or {}
   local nonseqs = options.nonsequences or {}
   local nsteps = options.nsteps or nil

   if nsteps == nil then
      nsteps = seqs[1].w:size(1)
      for i = 2, #seqs do
         local size
         if seqs[i].name == 'Tensor' then
            size = seqs[i].w:size(1)
         elseif type(seqs[i]) == 'table' then
            size = #seqs[i]
         else
            assert(false, 'sequence must be of type symtorch.Tensor or lua table')
         end

         if nsteps > size then
            nsteps = size
         end
      end
   end

   local function getinputs(i, src, dest)
      for i = 1, #src do
         local elem = src[i]
         if elem.name == 'Tensor' then
            table.insert(dest, elem)
         elseif type(elem) == 'table' then
            table.insert(dest, elem[i])
         else
            assert(false, 'sequence must be of type symtorch.Tensor or lua table')
         end
      end
   end

   local rv = {}
   for i = 1, nsteps do
      local inputs = {i}
      getinputs(i, seqs, inputs)
      getinputs(i, outputs, inputs)
      getinputs(i, nonseqs, inputs)

      local res = {fn(unpack(inputs))}
      table.insert(rv, res)
      outputs = res
   end
   return rv
end

return { scan = _scan }