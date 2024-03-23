import torch
import torch.nn as nn
import cupy

import halfkp


def _kernel_with_threads(kernel, n_threads):
    def f(grid, args):
        kernel(grid=grid, block=(n_threads,), args=args)
    return f


_fwd_kernel_cache = {}
def _make_forward_kernel(max_active_features, output_size):
    assert output_size <= 512, 'TODO: thread count for >= 512'
    n_threads = output_size

    key = (max_active_features, output_size)
    if key in _fwd_kernel_cache:
        return _fwd_kernel_cache[key]

    src = r'''
using uint32_t = unsigned int;
using int32_t = int;

extern "C" __global__
void feature_transformer_slice_forward(
        const int32_t* const feature_indices,
        const float*   const feature_values,
        const float*   const weight,
        const float*   const bias,
              float*   const output)
{
    const uint32_t max_active_features = ##max_active_features##;
    const uint32_t output_thread_slice_size = ##output_thread_slice_size##;
    const uint32_t output_size = ##output_size##;

    __shared__
        float shared_output[output_size];

    const uint32_t block_idx = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * output_thread_slice_size;

    float* const output_slice = output + block_idx * output_size + slice_offset;
    const float* bias_slice = bias + slice_offset;
    float* shared_output_slice = shared_output + slice_offset;

    const int32_t* feature_index_row = feature_indices + block_idx * max_active_features;
    const float* feature_value_row = feature_values + block_idx * max_active_features;

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        shared_output_slice[s] = bias_slice[s];

    for (uint32_t k = 0; k < max_active_features; ++k) {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];

        if (feature_index != -1) {
            const float* weight_slice = weight + feature_index * output_size + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < output_thread_slice_size; ++s)
                shared_output_slice[s] += weight_slice[s] * feature_value;
        } else break;
    }

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        output_slice[s] = shared_output_slice[s];
}

    '''.replace('##max_active_features##', str(max_active_features)) \
       .replace('##output_thread_slice_size##', str(output_size // n_threads)) \
       .replace('##output_size##', str(output_size))

    kernel = cupy.RawKernel(src, 'feature_transformer_slice_forward')
    kernel.compile()
    _fwd_kernel_cache[key] = _kernel_with_threads(kernel, n_threads)
    return _fwd_kernel_cache[key]


_bwd_kernel_cache = {}
def _make_backward_kernel(max_active_features, output_size):
    assert output_size <= 512, 'TODO: thread count for >= 512'
    n_threads = output_size

    key = (max_active_features, output_size)
    if key in _bwd_kernel_cache:
        return _bwd_kernel_cache[key]

    src = r'''
using uint32_t = unsigned int;
using int32_t = int;

extern "C" __global__
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad)
{
    const uint32_t max_active_features = ##max_active_features##;
    const uint32_t output_thread_slice_size = ##output_thread_slice_size##;
    const uint32_t output_size = ##output_size##;

    __shared__
        float shared_output_grad[output_size];

    const uint32_t block_idx = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * output_thread_slice_size;

    const float* output_grad_slice   = output_grad + block_idx * output_size + slice_offset;
          float* bias_grad_slice     = bias_grad + slice_offset;
          float* shared_output_grad_slice = shared_output_grad + slice_offset;

    const int32_t* feature_index_row = feature_indices + block_idx * max_active_features;
    const float* feature_value_row = feature_values + block_idx * max_active_features;

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        shared_output_grad_slice[s] = output_grad_slice[s];

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s) {
        const float sog = shared_output_grad_slice[s];
        if (sog != 0.f)
            atomicAdd(&bias_grad_slice[s], sog);
    }

    for (uint32_t k = 0; k < max_active_features; ++k) {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];

        if (feature_index != -1) {
            float* const weight_grad_slice = weight_grad 
                + feature_index * output_size + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < output_thread_slice_size; ++s) {
                const float sog = shared_output_grad_slice[s];
                if (sog != 0.f)
                    atomicAdd(&weight_grad_slice[s], sog * feature_value);
            }
        } else break;
    }

}
    '''.replace('##max_active_features##', str(max_active_features)) \
       .replace('##output_thread_slice_size##', str(output_size // n_threads)) \
       .replace('##output_size##', str(output_size))

    kernel = cupy.RawKernel(src, 'feature_transformer_slice_backward')
    kernel.compile()
    _bwd_kernel_cache[key] = _kernel_with_threads(kernel, n_threads)
    return _bwd_kernel_cache[key]


class FeatureTransformerSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ft_ics, ft_vals, weight, bias): # pyright: ignore
        ctx.save_for_backward(ft_ics, ft_vals, weight, bias)

        assert len(ft_ics.shape) == 2
        assert len(ft_vals.shape) == 2
        assert ft_ics.shape[0] == ft_vals.shape[0]
        assert ft_ics.shape[1] == ft_vals.shape[1]
        assert ft_ics.dtype == torch.int32
        assert ft_vals.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert ft_ics.is_cuda
        assert ft_vals.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert ft_vals.device == ft_ics.device
        assert weight.device == ft_ics.device
        assert bias.device == ft_ics.device

        assert ft_ics.is_contiguous()
        assert ft_vals.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = ft_ics.device
        batch_size = ft_ics.shape[0]
        max_active_features = ft_ics.shape[1]
        output_size = weight.shape[1]

        output = torch.empty((batch_size, output_size), dtype=torch.float32, 
                             device=device, requires_grad=True)

        kernel = _make_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                ft_ics.data_ptr(),
                ft_vals.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr(),
            )
        )

        return output

    @staticmethod
    def backward(ctx, grad_output): # pyright: ignore
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        ft_ics, ft_vals, weight, bias = ctx.saved_tensors

        batch_size = ft_ics.shape[0]
        max_active_features = ft_ics.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros_like(weight, requires_grad=False)
        bias_grad = torch.zeros_like(bias, requires_grad=False)

        kernel = _make_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            args=(
                ft_ics.data_ptr(),
                ft_vals.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
            )
        )

        return None, None, weight_grad, bias_grad


class FeatureTransformer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        s = n_in**-0.5
        self.weight = nn.Parameter(torch.rand(n_in, n_out) * (2 * s) - s)
        self.bias = nn.Parameter(torch.rand(n_out) * (2 * s) - s)

    @torch.no_grad()
    def coalesce_weights(self):
        assert self.n_in == halfkp.N_FT + halfkp.N_VIRT_FT
        self.weight.data = halfkp.coalesce_real_virtual_weights(self.weight.data)
        self.n_in = halfkp.N_FT

    @torch._dynamo.disable()
    def forward(self, ft_ics, ft_vals):
        return FeatureTransformerSliceFunction.apply(ft_ics, ft_vals, self.weight, self.bias)


