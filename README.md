# symtorch

I prefer [torch7](https://github.com/torch/torch7) over similar libraries for almost everything, but sometimes miss [theano](https://github.com/theano/theano)'s expressiveness, especially when constructing complex models like RNN's with LSTM units. symtorch brings expressive and trainable graph computation to torch. symtorch doesn't copy theano's symbolic tensors/computations because I'm not a huge fan of that.

Yes, I know [torch/nngraph](https://github.com/torch/nngraph) exists, but I think comparing LSTM examples in nngraph and symtorch will illustrate why I like symtorch.

LSTM's with nngraph, taken from [wojzaremba/lstm](https://github.com/wojzaremba/lstm/blob/master/main.lua#L65):

```lua
local function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end
```

Likewise, LSTM's with symtorch are written exactly like the equations tell us:

```lua
local symtorch = require 'symtorch'
local sigmoid = symtorch.sigmoid
local tanh = symtorch.tanh

local LSTM = Class {
   __init__ = function(self, input, hidden, output)
      self.input_size  = input
      self.hidden_size = hidden
      self.output_size = output

      self.W_xi = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hi = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_i  = symtorch.Tensor(hidden, 1)
      self.W_xf = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hf = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_f  = symtorch.Tensor(hidden, 1)
      self.W_xc = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_hc = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_c  = symtorch.Tensor(hidden, 1)
      self.W_xo = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_ho = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_o  = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.08) -- output decoder
      self.b_od = symtorch.Tensor(output, 1)

      self.prev_h = {}
      self.prev_c = {}
      for i = 1, input do
         table.insert(self.prev_h, symtorch.Tensor(hidden, 1))
         table.insert(self.prev_c, symtorch.Tensor(hidden, 1))
      end
   end,

   forward = function(self, input)
      -- Modeled after http://arxiv.org/pdf/1411.4555v1.pdf
      -- Equations 4-8

      local function step(i, x_t, prev_h, prev_c)
         local i_t = sigmoid(self.W_xi:dot(x_t) + self.W_hi:dot(prev_h) + self.b_i)                   -- (4)
         local f_t = sigmoid(self.W_xf:dot(x_t) + self.W_hf:dot(prev_h) + self.b_f)                   -- (5)
         local c_t = f_t * prev_c + i_t * tanh(self.W_xc:dot(x_t) + self.W_hc:dot(prev_h) + self.b_c) -- (6)
         local o_t = sigmoid(self.W_xo:dot(x_t) + self.W_ho:dot(prev_h) + self.b_o)                   -- (7)
         local h_t = o_t * tanh(c_t)                                                                  -- (8)
         self.prev_h[i] = h_t
         self.prev_c[i] = c_t
         return h_t, c_t
      end

      local mem = symtorch.Tensor(self.hidden_size, 1)
      local res = symtorch.scan{
         fn = step,
         sequences = {input, self.prev_h, self.prev_c}
      }
      local final = res[#res][1]
      return self.W_od:dot(final) + self.b_od
   end
}
```

This is still a work in progress, thus thoughts, feedback, and contributions are very welcome! I think the code should be more self-explanatory than the giant and impressive compiler theano. Convolutions, max pooling, and some other operations are written in C and called via LuaJIT FFI. I am currently writing a neural network library on top of symtorch, expect it to be similar to [benglard/Inveling/nn](https://github.com/benglard/Inveling/tree/master/nn) or [fchollet/keras](https://github.com/fchollet/keras).

## Installation

```
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/luaclass/master/luaclass-scm-1.rockspec
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/luaimport/master/luaimport-scm-1.rockspec
> (sudo) luarocks install https://raw.githubusercontent.com/benglard/symtorch/master/symtorch-scm-1.rockspec
```