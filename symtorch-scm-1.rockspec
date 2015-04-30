package = 'symtorch'
version = 'scm-1'

source = {
   url = 'git://github.com/benglard/symtorch'
}

description = {
   summary = 'Easy, expressive, trainable graph computation for torch',
   detailed = 'Easy, expressive, trainable graph computation for torch',
   homepage = 'https://github.com/benglard/symtorch'
}

dependencies = {
   'torch >= 7.0'
}

build = {
   type = 'command',
   build_command = '$(MAKE) LUA_BINDIR=$(LUA_BINDIR)  LUA_LIBDIR=$(LUA_LIBDIR)  LUA_INCDIR=$(LUA_INCDIR)',
   install_command = 'cp -r symtorch $(LUADIR)'
}