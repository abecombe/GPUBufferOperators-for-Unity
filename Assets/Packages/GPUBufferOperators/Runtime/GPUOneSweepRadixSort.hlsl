#ifndef CS_GPU_RADIX_SORT_HLSL
#define CS_GPU_RADIX_SORT_HLSL

//#pragma kernel InitHistogram
//#pragma kernel BuildHistogram
//#pragma kernel ScanHistogram
//#pragma kernel RadixSort

// default define
#ifndef DATA_TYPE
#define DATA_TYPE uint2  // input data struct
#endif
#ifndef GET_KEY
#define GET_KEY(s) s.x   // how to get the key-values used for sorting
#endif

#define NUM_GROUP_THREADS 128

StructuredBuffer<DATA_TYPE> data_in_buffer;
RWStructuredBuffer<DATA_TYPE> data_out_buffer;

RWStructuredBuffer<uint> first_index_buffer;
RWStructuredBuffer<uint> group_sum_buffer;
StructuredBuffer<uint> global_prefix_sum_buffer;
RWStructuredBuffer<uint> global_histogram_buffer;

uint num_elements;
uint num_groups;
uint group_offset;
uint bit_shift;
uint num_loops;
uint key_type;
uint sorting_order;

static const uint num_elements_per_group = NUM_GROUP_THREADS;
static const uint log_num_elements_per_group = log2(num_elements_per_group);
static const uint num_elements_per_group_minus_1 = num_elements_per_group - 1u;

static const uint n_way = 16u;
static const uint n_way_1 = n_way - 1u;
static const uint n_way_num_bits = 4u;
static const uint max_num_loops = 32u / n_way_num_bits;

static const uint s_data_len = num_elements_per_group;
static const uint s_scan_len = num_elements_per_group;
static const uint s_Pd_len = n_way;
static const uint s_histogram_len = n_way * max_num_loops;

groupshared DATA_TYPE s_data[s_data_len];
groupshared uint4 s_scan[s_scan_len];
groupshared uint s_Pd[s_Pd_len];
groupshared uint s_histogram[s_histogram_len];

inline uint float_to_uint_for_sorting(float f)
{
    uint mask = -(int)(asuint(f) >> 31) | 0x80000000;
    return asuint(f) ^ mask;
}
inline uint int_to_uint_for_sorting(int i)
{
    return asuint(i ^ 0x80000000);
}
inline uint get_key_4_bit(DATA_TYPE data)
{
    uint key;
    switch (key_type)
    {
        case 0: // uint
            key = GET_KEY(data);
            break;
        case 1: // int
            key = int_to_uint_for_sorting(GET_KEY(data));
            break;
        case 2: // float
            key = float_to_uint_for_sorting(GET_KEY(data));
            break;
        default:
            key = GET_KEY(data);
            break;
    }
    if (sorting_order == 1) // descending
    {
        key = ~key;
    }
    return (key >> bit_shift) & n_way_1;
}
inline uint get_key_4_bit(DATA_TYPE data, uint custom_bit_shift)
{
    uint key;
    switch (key_type)
    {
    case 0: // uint
        key = GET_KEY(data);
        break;
    case 1: // int
        key = int_to_uint_for_sorting(GET_KEY(data));
        break;
    case 2: // float
        key = float_to_uint_for_sorting(GET_KEY(data));
        break;
    default:
        key = GET_KEY(data);
        break;
    }
    if (sorting_order == 1) // descending
    {
        key = ~key;
    }
    return (key >> custom_bit_shift) & n_way_1;
}

inline uint get_value_in_uint16(uint4 uint16_value, uint key)
{
    return (uint16_value[key / 4u] >> (key % 4u * 8u)) & 0x000000ffu;
}
inline void set_value_in_uint16(inout uint4 uint16_value, uint value, uint key)
{
    uint16_value[key / 4u] += value << (key % 4u * 8u);
}
inline uint4 build_s_scan_data(uint key_4_bit)
{
    return (uint4)(key_4_bit / 4u == uint4(0u, 1u, 2u, 3u)) << ((key_4_bit % 4u) * 8u);
}

[numthreads(128, 1, 1)] // s_histogram_len
void InitHistogram(uint group_thread_id : SV_GroupThreadID)
{
    global_histogram_buffer[group_thread_id] = 0;
}

[numthreads(NUM_GROUP_THREADS, 1, 1)]
void BuildHistogram(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;
    uint global_id = num_elements_per_group * group_id + group_thread_id;

    DATA_TYPE data = (DATA_TYPE)0;
    if (global_id < num_elements)
    {
        data = data_in_buffer[global_id];
    }

    for (uint i = 0u; i < num_loops; i++)
    {
        uint key_4_bit = n_way_1;
        if (global_id < num_elements)
        {
            key_4_bit = get_key_4_bit(data, i * n_way_num_bits);
        }
        InterlockedAdd(s_histogram[key_4_bit + i * n_way], 1);
    }

    if (group_thread_id < num_loops * n_way)
    {
        InterlockedAdd(global_histogram_buffer[group_thread_id], s_histogram[group_thread_id]);
    }
}

[numthreads(128, 1, 1)] // s_histogram_len
void ScanHistogram(uint group_thread_id : SV_GroupThreadID)
{
    if (group_thread_id < num_loops * n_way)
    {
        s_histogram[group_thread_id] = global_histogram_buffer[group_thread_id];
    }
    GroupMemoryBarrierWithGroupSync();

    if (group_thread_id < num_loops)
    {
        uint scan_data = 0u;
        for (uint i = 0u; i < n_way; i++)
        {
            uint temp = s_histogram[group_thread_id * n_way + i];
            s_histogram[group_thread_id * n_way + i] = scan_data;
            scan_data += temp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (group_thread_id < num_loops * n_way)
    {
        global_histogram_buffer[group_thread_id] = s_histogram[group_thread_id];
    }
}

[numthreads(NUM_GROUP_THREADS, 1, 1)]
void RadixSort(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;
    uint global_id = num_elements_per_group * group_id + group_thread_id;

    // extract 4 bits
    DATA_TYPE data = (DATA_TYPE)0;
    uint key_4_bit = n_way_1;
    if (global_id < num_elements)
    {
        data = data_in_buffer[global_id];
        key_4_bit = get_key_4_bit(data);
    }

    // build scan data
    s_scan[group_thread_id] = build_s_scan_data(key_4_bit);
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele scan
    [unroll(log_num_elements_per_group)]
    for (uint offset = 1u;; offset <<= 1u)
    {
        uint4 sum = s_scan[group_thread_id];
        if (group_thread_id >= offset)
        {
            sum += s_scan[group_thread_id - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        s_scan[group_thread_id] = sum;
        GroupMemoryBarrierWithGroupSync();
    }

    // calculate first index of each 4bit key-value
    uint4 total = s_scan[num_elements_per_group_minus_1];
    uint4 first_index = 0u;
    uint run_sum = 0u;
    [unroll(n_way)]
    for (uint i = 0u;; ++i)
    {
        set_value_in_uint16(first_index, run_sum, i);
        run_sum += get_value_in_uint16(total, i);
    }

    if (group_thread_id < n_way)
    {
        // copy sums of each 4bit key-value to global memory
        group_sum_buffer[group_thread_id * num_groups + group_id] = get_value_in_uint16(total, group_thread_id);
        // copy first index of each 4bit key-value to global memory
        first_index_buffer[group_thread_id + n_way * group_id] = get_value_in_uint16(first_index, group_thread_id);
    }

    // sort the input data locally
    uint new_id = get_value_in_uint16(first_index, key_4_bit);
    if (group_thread_id > 0u)
    {
        new_id += get_value_in_uint16(s_scan[group_thread_id - 1u], key_4_bit);
    }
    s_data[new_id] = data;
    GroupMemoryBarrierWithGroupSync();

    // copy sorted input data to global memory
    if (global_id < num_elements)
    {
        data_out_buffer[global_id] = s_data[group_thread_id];
    }
}

/**
 * \brief copy input data to final position in global memory
 */
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void GlobalShuffle(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;
    uint global_id = num_elements_per_group * group_id + group_thread_id;

    if (group_thread_id < n_way)
    {
        // set global destination of each 4bit key-value
        s_Pd[group_thread_id] = global_prefix_sum_buffer[group_thread_id * num_groups + group_id];
        // subtract the first index of each 4bit key-value
        // to make it easier to calculate the final destination of individual data
        s_Pd[group_thread_id] -= first_index_buffer[group_thread_id + n_way * group_id];
    }
    GroupMemoryBarrierWithGroupSync();

    if (global_id < num_elements)
    {
        DATA_TYPE data = data_in_buffer[global_id];
        uint key_4_bit = get_key_4_bit(data);

        uint new_id = group_thread_id + s_Pd[key_4_bit];

        // copy data to the final destination
        data_out_buffer[new_id] = data;
    }
}


#endif /* CS_GPU_RADIX_SORT_HLSL */