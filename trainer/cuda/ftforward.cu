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
    const uint32_t max_active_features = 30;
    const uint32_t output_thread_slice_size = 1;
    const uint32_t output_size = 256;

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

