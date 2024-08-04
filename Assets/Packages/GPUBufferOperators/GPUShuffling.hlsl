#ifndef CS_GPU_SHUFFLE_HLSL
#define CS_GPU_SHUFFLE_HLSL

//#pragma kernel ApplyBijectiveFunction
//#pragma kernel ShuffleElements
//#pragma kernel CopyBuffer
//#pragma use_dxc

// default define
#ifndef DATA_TYPE
#define DATA_TYPE uint2  // input data struct
#endif

#define NUM_GROUP_THREADS 128

#define ulong uint64_t

StructuredBuffer<DATA_TYPE> data_in_buffer;
RWStructuredBuffer<DATA_TYPE> data_out_buffer;
StructuredBuffer<uint> bijection_shuffle_buffer_read;
RWStructuredBuffer<uint> bijection_shuffle_buffer_write;
StructuredBuffer<uint> frag_scan_buffer_read;
RWStructuredBuffer<uint> frag_scan_buffer_write;

uint num_elements;
uint num_pow_of_2_elements;
uint group_offset;

uint right_side_bits;
uint left_side_bits;
uint right_side_mask;
uint left_side_mask;

uint num_rounds;
uint4 key;

static const uint num_elements_per_group = NUM_GROUP_THREADS;

static const ulong M0 = 0xD2B74407B1CE6E93;

inline void mul_high_low(ulong a, uint b, out uint high_p, out uint low_p)
{
    ulong product = a * (ulong)b;
    high_p = (uint)(product >> 32u);
    low_p = (uint)product;
}
inline uint variable_philox(uint input)
{
    uint2 state = uint2(input >> right_side_bits, input & right_side_mask);
    for (uint i = 0; i < num_rounds; i++)
    {
        uint high, low;
        mul_high_low(M0, state.x, high, low);
        low = low << (right_side_bits - left_side_bits) | state.y >> left_side_bits;
        state.x = ((high ^ key[i]) ^ state.y) & left_side_mask;
        state.y = low & right_side_mask;
    }
    return (state.x << right_side_bits) | state.y;
}
inline uint bijective_function(uint input)
{
    return variable_philox(input);
}

/**
 * \brief apply bijective function to [0, num_pow_of_2_elements] and output the result
 */
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void ApplyBijectiveFunction(uint thread_id : SV_DispatchThreadID)
{
    thread_id += group_offset * num_elements_per_group;

    if (thread_id < num_pow_of_2_elements)
    {
        uint W = bijective_function(thread_id);
        bijection_shuffle_buffer_write[thread_id] = W;
        frag_scan_buffer_write[thread_id] = W < num_elements ? 1u : 0u;
    }
}

/**
 * \brief shuffle elements based on the bijection_shuffle_buffer
 */
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void ShuffleElements(uint thread_id : SV_DispatchThreadID)
{
    thread_id += group_offset * num_elements_per_group;

    if (thread_id < num_pow_of_2_elements)
    {
        uint W = bijection_shuffle_buffer_read[thread_id];
        if (W < num_elements)
        {
            uint frag_scan = frag_scan_buffer_read[thread_id];
            data_out_buffer[frag_scan] = data_in_buffer[W];
        }
    }
}

/**
 * \brief copy input buffer to output buffer
 */
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void CopyBuffer(uint thread_id : SV_DispatchThreadID)
{
    thread_id += group_offset * num_elements_per_group;

    if (thread_id < num_elements)
    {
        data_out_buffer[thread_id] = data_in_buffer[thread_id];
    }
}

#endif /* CS_GPU_SHUFFLE_HLSL */