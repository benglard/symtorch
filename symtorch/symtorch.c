#include <math.h>
#include <stdbool.h>

void tensor_tanh_backward(
  double* input_dw,
  double* output_w,
  double* output_dw,
  const unsigned int size
) {
  double ow = 0.0;
  for(int i = 0; i < size; i++) {
    ow = output_w[i];
    input_dw[i] += (1.0 - ow * ow) * output_dw[i];
  }
}

void tensor_sigmoid(double* tensor, const unsigned int size) {
  for(int i = 0; i < size; i++) {
    tensor[i] = 1.0 / (1.0 + exp(-tensor[i]));
  }
}

void tensor_sigmoid_backward(
  double* input_dw,
  double* output_w,
  double* output_dw,
  const unsigned int size
) {
  double ow = 0.0;
  for(int i = 0; i < size; i++) {
    ow = output_w[i];
    input_dw[i] += ow * (1.0 - ow) * output_dw[i];
  }
}

void tensor_relu(double* tensor, const unsigned int size) {
  for(int i = 0; i < size; i++) {
    tensor[i] = tensor[i] > 0.0 ? tensor[i] : 0.0;
  }
}

void tensor_relu_backward(
  double* input_dw,
  double* output_w,
  double* output_dw,
  const unsigned int size
) {
  for(int i = 0; i < size; i++) {
    input_dw[i] += output_w[i] <= 0.0 ? 0.0 : output_dw[i];
  }
}

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
) {
  unsigned int d, ax, ay, fy, fx, fd, idx1, idx2;
  int x, y, off_x, off_y;
  double sum;

  for(d=0; d < out_depth; d++) {
    x = -pad;
    y = -pad;
    for(ay=0; ay < out_sy; y+=stride, ay++) {
      x = -pad;
      for(ax=0; ax < out_sx; x+=stride, ax++) {
        // convolve centered at this particular location
        sum = 0.0;
        for(fy=0; fy < fsy; fy++) {
          off_y = y + fy;
          for(fx=0; fx < fsx; fx++) {
            off_x = x + fx;
            if (off_y >= 0 && off_y < in_sy &&
                off_x >= 0 && off_x < in_sx) {
              for(fd=0; fd < in_depth; fd++) {
                idx1 = ((fsx * fy) + fx) * in_depth + fd;
                idx2 = ((in_sx * off_y) + off_x) * in_depth + fd;
                sum += filter[idx1] * input[idx2];
              }
            }
          }
        }
        sum += use_bias ? bias[d] : 0.0;
        output[((out_sx * ay) + ax) * out_depth + d] = sum;
      }
    }
  }
}

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
) {
  unsigned int d, ax, ay, fy, fx, fd, idx1, idx2;
  int x, y, off_x, off_y;
  double grad;

  for(d=0; d < out_depth; d++) {
    x = -pad;
    y = -pad;
    for(ay=0; ay < out_sy; y+=stride, ay++) {
      x = -pad;
      for(ax=0; ax < out_sx; x+=stride, ax++) {
        // convolve and add up the gradients
        // gradient from above, from chain rule
        grad = output_dw[((out_sx * ay) + ax) * out_depth + d];

        for(fy=0; fy < fsy; fy++) {
          off_y = y + off_y;
          for(fx=0; fx < fsx; fx++) {
            off_x = x + fx;
            if (off_y >= 0 && off_y < in_sy &&
                off_x >= 0 && off_x < in_sx) {
              for(fd=0; fd < in_depth; fd++) {
                idx1 = ((in_sx * off_y) + off_x) * in_depth + fd;
                idx2 = ((fsx * fy) + fx) * in_depth + fd;
                filter_dw[idx2] += input_w[idx1] * grad;
                input_dw[idx1] += filter_w[idx2] * grad;
              }
            }
          }
        }

        if (use_bias) {
          bias_dw[d] += grad;
        }
      }
    }
  }
}

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
) {
  unsigned int d, ax, ay, fy, fx;
  int x, y, off_x, off_y, n = 0;
  double win_x, win_y, act, max_act;

  for(d=0; d < out_depth; d++) {
    x = -pad;
    y = -pad;
    for(ax=0; ax < out_sx; x+=stride, ax++) {
      y = -pad;
      for(ay=0; ay < out_sy; y+=stride, ay++) {
        // convolve centered at this particular location
        win_x = -1;
        win_y = -1;
        max_act = -99999;

        for(fx=0; fx < fsx; fx++) {
          for(fy=0; fy < fsy; fy++) {
            off_x = x + fx;
            off_y = y + fy;

            if (off_y >= 0 && off_y < in_sy &&
                off_x >= 0 && off_x < in_sx) {
              act = input[((in_sx * off_y) + off_x) * in_depth + d];
              if (act > max_act) {
                max_act = act;
                win_x = off_x;
                win_y = off_y;
              }
            }
          }
        }

        x_windows[n] = win_x;
        y_windows[n] = win_y;
        n++;
        output[((out_sx * ay) + ax) * out_depth + d] = max_act;
      }
    }
  }
}

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
) {
  // pooling layers have no parameters, so simply compute
  // gradient wrt data here
  unsigned int d, ax, ay;
  int x, y, off_x, off_y, n = 0;
  double grad;

  for(d=0; d < out_depth; d++) {
    x = -pad;
    y = -pad;
    for(ax=0; ax < out_sx; x+=stride, ax++) {
      y = -pad;
      for(ay=0; ay < out_sy; y+=stride, ay++) {
        grad = output_dw[((out_sx * ay) + ax) * out_depth + d];
        off_x = (int)x_windows[n];
        off_y = (int)y_windows[n];
        input_dw[((in_sx * off_y) + off_x) * in_depth + d] += grad;
        n++;
      }
    }
  }
}