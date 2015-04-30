require 'torch'
require 'luaclass'
require 'luaimport'

local path = debug.getinfo(1).source:sub(2)
symtorch = { cpath = paths.concat(path:match('(.*/)'), 'libsymtorch.o') }
package 'symtorch'
import {
   'graph',
   'tensor',
   'ops',
   'conv',
   'random'
}
_graph = symtorch.Graph()
return symtorch