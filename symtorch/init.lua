require 'torch'
require 'luaclass'
require 'luaimport'

local path = debug.getinfo(1).source:sub(2)
symtorch = { cpath = paths.concat(path:match('(.*/)'), 'libsymtorch.so') }
symtorch = Package {
   'graph',
   'tensor',
   'ops',
   'conv',
   'random',
   'scan',
   'update'
}
_graph = symtorch.Graph()
return symtorch