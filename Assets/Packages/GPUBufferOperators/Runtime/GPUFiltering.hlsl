#ifndef CS_GPU_FILTERING_HLSL
#define CS_GPU_FILTERING_HLSL

//#pragma kernel RadixSortLocal
//#pragma kernel GlobalShuffle

// default define
#ifndef DATA_TYPE
#define DATA_TYPE uint2        // input data struct
#endif
#ifndef GET_KEY
#define GET_KEY(s) (s.x == 1)  // certain condition used for filtering
#endif

#define NUM_GROUP_THREADS 128

StructuredBuffer<DATA_TYPE> data_in_buffer;
RWStructuredBuffer<DATA_TYPE> data_out_buffer;

RWStructuredBuffer<uint> group_sum_buffer;
RWStructuredBuffer<uint> global_prefix_sum_buffer;

uint num_elements;
uint num_groups;
uint group_offset;

static const uint num_elements_per_group = NUM_GROUP_THREADS;
static const uint log_num_elements_per_group = log2(num_elements_per_group);
static const uint num_elements_per_group_minus_1 = num_elements_per_group - 1u;

static const uint s_data_len = num_elements_per_group;
static const uint s_scan_len = num_elements_per_group;

groupshared DATA_TYPE s_data[s_data_len];
groupshared uint s_scan[s_scan_len];

/**
 * \brief sort input data locally and output the number of filtered elements within groups
 */
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void RadixSortLocal(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;
    uint global_id = num_elements_per_group * group_id + group_thread_id;

    // extract key
    DATA_TYPE data;
    bool key = false;
    if (global_id < num_elements)
    {
        data = data_in_buffer[global_id];
        key = GET_KEY(data);
    }

    // build scan data
    s_scan[group_thread_id] = (uint)key;
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele scan
    [unroll(log_num_elements_per_group)]
    for (uint offset = 1u;; offset <<= 1u)
    {
        uint sum = s_scan[group_thread_id];
        if (group_thread_id >= offset)
        {
            sum += s_scan[group_thread_id - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        s_scan[group_thread_id] = sum;
        GroupMemoryBarrierWithGroupSync();
    }

    uint total = s_scan[num_elements_per_group_minus_1];

    if (group_thread_id == 0u)
    {
        // copy the number of filtered elements to global memory
        group_sum_buffer[group_id] = total;
        global_prefix_sum_buffer[group_id] = total;
    }

    // sort the input data locally
    uint new_id = 0u;
    if (group_thread_id > 0u)
    {
        new_id = s_scan[group_thread_id - 1u];
    }
    if (key)
    {
        s_data[new_id] = data;
    }
    GroupMemoryBarrierWithGroupSync();

    // copy sorted input data to global memory
    if (group_thread_id < total)
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

    if (group_thread_id < group_sum_buffer[group_id])
    {
        uint new_id = group_thread_id + global_prefix_sum_buffer[group_id];
        // copy data to the final destination
        data_out_buffer[new_id] = data_in_buffer[global_id];
    }
}


#endif /* CS_GPU_FILTERING_HLSL */