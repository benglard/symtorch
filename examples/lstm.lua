-- https://github.com/benglard/Inveling/blob/master/nn/lstm.py
local symtorch = require '../symtorch'
local sigmoid = symtorch.sigmoid
local tanh = symtorch.tanh

local LSTM = Class {
   __init__ = function(self, input, hidden, output)
      self.input_size  = input
      self.hidden_size = hidden
      self.output_size = output

      self.W_hi = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_ci = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_i  = symtorch.Tensor(hidden, 1)
      self.W_hf = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_cf = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_f  = symtorch.Tensor(hidden, 1)
      self.W_hc = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.b_c  = symtorch.Tensor(hidden, 1)
      self.W_ho = symtorch.Tensor(hidden, input):rand(0, 0.08)
      self.W_co = symtorch.Tensor(hidden, hidden):rand(0, 0.08)
      self.b_o  = symtorch.Tensor(hidden, 1)
      self.W_od = symtorch.Tensor(output, hidden):rand(0, 0.08) -- output decoder
      self.b_od = symtorch.Tensor(output, 1)
   end,

   forward = function(self, x)
      -- Modeled after
      -- https://www.cs.toronto.edu/~hinton/absps/RNN13.pdf
      -- Equations 3-7

      local function step(prev_h, prev_c)
         local i_t = sigmoid(self.W_hi:dot(prev_h) + self.W_ci:dot(prev_c) + self.b_i) -- (3)
         local f_t = sigmoid(self.W_hf:dot(prev_h) + self.W_cf:dot(prev_c) + self.b_f) -- (4)
         local c_t = f_t * prev_c + i_t * tanh(self.W_hc:dot(prev_h) + self.b_c)       -- (5)
         local o_t = sigmoid(self.W_ho:dot(prev_h) + self.W_co:dot(prev_c) + self.b_o) -- (6)
         local h_t = o_t * tanh(c_t)                                                   -- (7)
         return h_t, c_t
      end

      local mem = symtorch.Tensor(self.hidden_size, 1)
      local res = symtorch.scan{fn=step, sequences={x, mem}, nsteps=1}
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