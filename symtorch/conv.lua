local ffi = require 'ffi'
local libsymtorch = ffi.load(symtorch.cpath)
ffi.cdef[[
void tensor_conv2d(
  double* input,
  double* output,
  double* filter,
  unsigned int in_depth,
  unsigned int in_sx,
  unsigned int in_sy,
  unsigned int out_depth,
  unsigned int out_sx,
  unsigned int out_sy,
  unsigned int fsx,
  unsigned int fsy,
  int stride,
  int pad
);

void tensor_conv2d_backward(
  double* input_w,
  double* input_dw,
  double* output_w,
  double* output_dw,
  double* filter_w,
  double* filter_dw,
  unsigned int in_depth,
  unsigned int in_sx,
  unsigned int in_sy,
  unsigned int out_depth,
  unsigned int out_sx,
  unsigned int out_sy,
  unsigned int fsx,
  unsigned int fsy,
  int stride,
  int pad
);

void tensor_maxpool2d(
  double* input,
  double* output,
  double* x_windows,
  double* y_windows,
  unsigned int in_depth,
  unsigned int in_sx,
  unsigned int in_sy,
  unsigned int out_depth,
  unsigned int out_sx,
  unsigned int out_sy,
  unsigned int fsx,
  unsigned int fsy,
  int stride,
  int pad
);

void tensor_maxpool2d_backward(
  double* input_w,
  double* input_dw,
  double* output_w,
  double* output_dw,
  double* x_windows,
  double* y_windows,
  unsigned int in_depth,
  unsigned int in_sx,
  unsigned int in_sy,
  unsigned int out_depth,
  unsigned int out_sx,
  unsigned int out_sy,
  unsigned int fsx,
  unsigned int fsy,
  int stride,
  int pad
);
]]

local _conv2 = function(input, filter, stride, pad)
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

   -- convolve
   libsymtorch.tensor_conv2d(
      input.w:data(),
      output.w:data(),
      filter.w:data(),
      in_depth, in_sx, in_sy,
      out_depth, out_sx, out_sy,
      sx, sy, stride, pad)

   _graph:add(function()
      libsymtorch.tensor_conv2d_backward(
         input.w:data(), input.dw:data(),
         output.w:data(), output.dw:data(),
         filter.w:data(), filter.dw:data(),
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