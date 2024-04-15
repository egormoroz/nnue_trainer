using uint32_t = unsigned int;
using int32_t = int;

extern "C" __global__
void psqt_slice_forward(
        const int32_t* const feature_indices,
        const float*   const weight,
              float*   const output)
{
    const uint32_t max_active_features = 32;
    const uint32_t pos_per_block = 1;

    const uint32_t pos_idx = blockIdx.x * pos_per_block + threadIdx.x;
    const int32_t* feature_index_row = feature_indices + pos_idx * max_active_features;

    // __shared__
    //     float cached_weight[n_features];

    float psqt = 0;
    for (uint32_t i = 0; i < max_active_features; ++i) {
        const int32_t feature_index = feature_index_row[i];
        if (feature_index != -1)
            psqt += weight[feature_index];
        else break;
    }

    output[pos_idx] = psqt;
}

extern "C" __global__
void psqt_slice_backward(
        const int32_t* const feature_indices,
        const float*   const weight,
              float*   const weight_grad,
        const float*   const output_grad)
{
    const uint32_t max_active_features = 32;
    const uint32_t pos_per_block = 1;

    const uint32_t pos_idx = blockIdx.x * pos_per_block + threadIdx.x;
    const int32_t* feature_index_row = feature_indices + pos_idx * max_active_features;

    const float sog = output_grad[pos_idx];

    float psqt = 0;
    for (uint32_t i = 0; i < max_active_features; ++i) {
        const int32_t feature_index = feature_index_row[i];

        if (feature_index != -1)
            atomicAdd(&weight_grad[feature_index], sog);
        else break;
    }

}

