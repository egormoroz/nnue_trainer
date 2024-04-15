using uint32_t = unsigned int;
using int32_t = int;

extern "C" __global__
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad)
{
    const uint32_t max_active_features = 30;
    const uint32_t output_thread_slice_size = 1;
    const uint32_t output_size = 256;
    const uint32_t out_size = output_size + 1;

    __shared__
        float shared_output_grad[output_size];

    const uint32_t block_idx = blockIdx.x;
    const uint32_t slice_offset = threadIdx.x * output_thread_slice_size;

    const float* output_grad_slice   = output_grad + block_idx * out_size + slice_offset + 1;
          float* bias_grad_slice     = bias_grad + slice_offset;
          float* shared_output_grad_slice = shared_output_grad + slice_offset;

    const int32_t* feature_index_row = feature_indices + block_idx * max_active_features;

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

        if (feature_index != -1) {
            float* const weight_grad_slice = weight_grad 
                + feature_index * out_size + slice_offset + 1;
            #pragma unroll
            for (uint32_t s = 0; s < output_thread_slice_size; ++s) {
                const float sog = shared_output_grad_slice[s];
                if (sog != 0.f)
                    atomicAdd(&weight_grad_slice[s], sog);
            }
        } else break;
    }

    if (threadIdx.x == 0) {
        const float sog = output_grad_slice[0];

        for (uint32_t k = 0; k < max_active_features; ++k) {
            const int32_t feature_index = feature_index_row[k];
            if (feature_index != -1) {
                atomicAdd(&weight_grad[feature_index * out_size], sog);
            } else break;
        }
    }
}

