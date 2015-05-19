local ffi = require 'ffi'
local libsymtorch = ffi.load(symtorch.cpath)
ffi.cdef[[
void tensor_conv2d(
  const double* input,
  double* output,
  const double* filter,
  const bool use_bias,
  const double* bias,
  const unsigned int in_depth,
  const unsigned int in_sx,
  const unsigned int in_sy,
  const unsigned int out_depth,
  const unsigned int out_sx,
  const unsigned int out_sy,
  const unsigned int fsx,
  const unsigned int fsy,
  const int stride,
  const int pad
);
void tensor_conv2d_backward(
  const double* input_w,
  double* input_dw,
  const double* output_w,
  double* output_dw,
  const double* filter_w,
  double* filter_dw,
  const bool use_bias,
  double* bias_dw,
  const unsigned int in_depth,
  const unsigned int in_sx,
  const unsigned int in_sy,
  const unsigned int out_depth,
  const unsigned int out_sx,
  const unsigned int out_sy,
  const unsigned int fsx,
  const unsigned int fsy,
  const int stride,
  const int pad
);

void tensor_maxpool2d(
  const double* input,
  double* output,
  double* x_windows,
  double* y_windows,
  const unsigned int in_depth,
  const unsigned int in_sx,
  const unsigned int in_sy,
  const unsigned int out_depth,
  const unsigned int out_sx,
  const unsigned int out_sy,
  const unsigned int fsx,
  const unsigned int fsy,
  const int stride,
  const int pad
);

void tensor_maxpool2d_backward(
  const double* input_w,
  double* input_dw,
  const double* output_w,
  const double* output_dw,
  const double* x_windows,
  const double* y_windows,
  const unsigned int in_depth,
  const unsigned int in_sx,
  const unsigned int in_sy,
  const unsigned int out_depth,
  const unsigned int out_sx,
  const unsigned int out_sy,
  const unsigned int fsx,
  const unsigned int fsy,
  const int stride,
  const int pad
);
]]

local _conv2 = function(input, filter, stride, pad, bias)
   assert(input.w:dim() == 3 and filter.w:dim() == 3,
      'Only 3D tensors are allowed as input to symtorch.conv2d as of now.')

   -- input size
   local in_depth = input.w:size(1)
   local in_sx = input.w:size(2)
   local in_sy = input.w:size(3)

   -- filter size
   local sx = filter.w:size(2) 
   local sy = filter.w:size(3) or sx

   stride = stride or 1
   pad = pad or 0

   -- output size
   local out_depth = filter.w:size(1)
   local out_sx = math.floor((in_sx - sx + 2 * pad) / stride + 1)
   local out_sy = math.floor((in_sy - sy + 2 * pad) / stride + 1)
   local output = symtorch.Tensor(out_depth, out_sx, out_sy)

   local use_bias = bias ~= nil
   if use_bias then
      if type(bias) == 'number' then
         bias = symtorch.Tensor(out_depth):fill(bias)
      elseif bias:isTensor() then
         bias = symtorch.Tensor(out_depth):copy(bias)
      elseif bias.name == 'Tensor' then
         bias:resize(out_depth)
      else
         assert(false, 'Bias must be of type number | torch.Tensor | symtorch.Tensor')
      end
   else
      bias = symtorch.Tensor(0)
   end

   -- convolve
   libsymtorch.tensor_conv2d(
      input.w:data(),
      output.w:data(),
      filter.w:data(),
      use_bias, bias.w:data(),
      in_depth, in_sx, in_sy,
      out_depth, out_sx, out_sy,
      sx, sy, stride, pad)

   _graph:add(function()
      libsymtorch.tensor_conv2d_backward(
         input.w:data(), input.dw:data(),
         output.w:data(), output.dw:data(),
         filter.w:data(), filter.dw:data(),
         use_bias, bias.dw:data(),
         in_depth, in_sx, in_sy,
         out_depth, out_sx, out_sy,
         sx, sy, stride, pad)
   end)

   return output
end

local _pool = function(input, sx, sy, stride, pad)
   assert(input.w:dim() == 3,
      'Only 3D tensors are allowed as input to symtorch.maxpool2d as of now.')

   -- input size
   local in_depth = input.w:size(1)
   local in_sx = input.w:size(2)
   local in_sy = input.w:size(3)

   sx = sx or 2
   sy = sy or 2
   stride = stride or 2
   pad = pad or 0

   -- output size
   local out_depth = in_depth
   local out_sx = math.floor((in_sx - sx + 2 * pad) / stride + 1)
   local out_sy = math.floor((in_sy - sy + 2 * pad) / stride + 1)
   local output = symtorch.Tensor(out_depth, out_sx, out_sy)

   -- window storage (filter)
   local x_windows = symtorch.Tensor(out_depth, out_sx, out_sy)
   local y_windows = symtorch.Tensor(out_depth, out_sx, out_sy)

   -- pool
   libsymtorch.tensor_maxpool2d(
      input.w:data(),
      output.w:data(),
      x_windows.w:data(),
      y_windows.w:data(),
      in_depth, in_sx, in_sy,
      out_depth, out_sx, out_sy,
      sx, sy, stride, pad)

   _graph:add(function()
      libsymtorch.tensor_maxpool2d_backward(
         input.w:data(), input.dw:data(),
         output.w:data(), output.dw:data(),
         x_windows.w:data(), y_windows.w:data(),
         in_depth, in_sx, in_sy,
         out_depth, out_sx, out_sy,
         sx, sy, stride, pad)
   end)

   return output
end

return {
   conv2d = _conv2,
   maxpool2d = _pool
}