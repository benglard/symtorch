local elemApply = function(elem, cb1, cb2, ...)
   if elem.name == 'Tensor' then
      return cb1(elem, ...)
   elseif type(elem) == 'table' then
      return cb2(elem, ...)
   else
      assert(false, 'sequence must be of type symtorch.Tensor or lua table')
   end
end

local _scan = function(options)
   options = options or {}
   local fn = options.fn or function(...) end
   local seqs = options.sequences or {}
   local outputs = options.outputs or {}
   local nonseqs = options.nonsequences or {}
   local nsteps = options.nsteps or nil

   local tensorSize = function(e) return e.w:size(1) end
   local tableSize = function(e) return #e end

   if nsteps == nil then
      nsteps = elemApply(seqs[1], tensorSize, tableSize)
      for i = 2, #seqs do
         local size = elemApply(seqs[i], tensorSize, tableSize)
         if nsteps > size then
            nsteps = size
         end
      end
   end

   local identity = function(e) return e end
   local getElem = function(e, ...) return e[...] end
   local function getinputs(i, src, dest)
      for j = 1, #src do
         local elem = src[j]
         table.insert(dest, elemApply(elem, identity, getElem, i))
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