-- https://github.com/benglard/Inveling/blob/master/nn/lstm.py
local symtorch = require '../symtorch'
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

      local res = symtorch.scan{
         fn = step,
         sequences = {input, self.prev_h, self.prev_c}
      }
      local final = res[#res][1]
      return self.W_od:dot(final) + self.b_od
   end
}

local lstm = LSTM(10, 20, 2)
local x1 = symtorch.Tensor(10, 1):rand(0, 1)
local x2 = symtorch.Tensor(10, 1):rand(0, 1)
local x3 = symtorch.Tensor(10, 1):rand(0, 1)
local o1 = lstm:forward(x1)
local o2 = lstm:forward(x2)
local o3 = lstm:forward(x3)

print(o1.w)
print(o2.w)
print(o3.w)