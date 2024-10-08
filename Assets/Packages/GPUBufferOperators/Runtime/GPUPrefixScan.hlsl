﻿#ifndef CS_GPU_PREFIX_SCAN_HLSL
#define CS_GPU_PREFIX_SCAN_HLSL

//#pragma kernel PrefixScan
//#pragma kernel AddGroupSum
//#pragma multi_compile _ NUM_GROUP_THREADS_128 NUM_GROUP_THREADS_256 NUM_GROUP_THREADS_512
//#pragma multi_compile _ DATA_TYPE_UINT DATA_TYPE_INT DATA_TYPE_FLOAT

#if !defined(NUM_GROUP_THREADS_128) && !defined(NUM_GROUP_THREADS_256) && !defined(NUM_GROUP_THREADS_512)
#define NUM_GROUP_THREADS_128
#endif

#if defined(NUM_GROUP_THREADS_128)
#define NUM_GROUP_THREADS 128
#elif defined(NUM_GROUP_THREADS_256)
#define NUM_GROUP_THREADS 256
#elif defined(NUM_GROUP_THREADS_512)
#define NUM_GROUP_THREADS 512
#endif

#if !defined(DATA_TYPE_UINT) && !defined(DATA_TYPE_INT) && !defined(DATA_TYPE_FLOAT)
#define DATA_TYPE_UINT
#endif

#if defined(DATA_TYPE_UINT)
#define DATA_TYPE uint
#elif defined(DATA_TYPE_INT)
#define DATA_TYPE int
#elif defined(DATA_TYPE_FLOAT)
#define DATA_TYPE float
#endif

// macro used for computing bank-conflict-free shared memory array indices
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define SHARED_MEMORY_ADDRESS(n) ((n) + CONFLICT_FREE_OFFSET(n))

RWStructuredBuffer<DATA_TYPE> data_buffer;
RWStructuredBuffer<DATA_TYPE> group_sum_buffer;

uint num_elements;
uint group_offset;
uint group_sum_offset;
bool is_inclusive_scan;

static const uint num_group_threads = NUM_GROUP_THREADS;
static const uint num_elements_per_group = 2u * NUM_GROUP_THREADS;
static const uint num_elements_per_group_minus_1 = num_elements_per_group - 1u;
static const uint sma_num_elements_per_group = SHARED_MEMORY_ADDRESS(num_elements_per_group);
static const uint sma_num_elements_per_group_minus_1 = SHARED_MEMORY_ADDRESS(num_elements_per_group_minus_1);
static const uint log_num_elements_per_group = log2(num_elements_per_group);

static const uint s_scan_len = sma_num_elements_per_group + 1u;

groupshared DATA_TYPE s_scan[s_scan_len];

// scan input data locally and output total sums within groups
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void PrefixScan(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;

    // handle two values in one thread
    uint ai = group_thread_id;
    uint bi = ai + num_group_threads;
    ai = SHARED_MEMORY_ADDRESS(ai);
    bi = SHARED_MEMORY_ADDRESS(bi);

    uint global_ai = group_thread_id + num_elements_per_group * group_id;
    uint global_bi = global_ai + num_group_threads;

    // copy input data to shared memory
    s_scan[ai] = global_ai < num_elements ? data_buffer[global_ai] : 0;
    s_scan[bi] = global_bi < num_elements ? data_buffer[global_bi] : 0;

    uint offset = 1u;

    // upsweep step
    [unroll(log_num_elements_per_group)]
    for (uint du = num_elements_per_group >> 1u;; du >>= 1u)
    {
        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < du)
        {
            uint ai_u = offset * ((group_thread_id << 1u) + 1u) - 1u;
            uint bi_u = offset * ((group_thread_id << 1u) + 2u) - 1u;
            ai_u = SHARED_MEMORY_ADDRESS(ai_u);
            bi_u = SHARED_MEMORY_ADDRESS(bi_u);

            s_scan[bi_u] += s_scan[ai_u];
        }

        offset <<= 1u;
    }

    // save the total sum on global memory
    if (group_thread_id == 0u)
    {
        DATA_TYPE group_sum = s_scan[sma_num_elements_per_group_minus_1];
        group_sum_buffer[group_id + group_sum_offset] = group_sum;
        s_scan[sma_num_elements_per_group_minus_1] = 0;
        s_scan[sma_num_elements_per_group] = group_sum;
    }

    // downsweep step
    [unroll(log_num_elements_per_group)]
    for (uint dd = 1u;; dd <<= 1u)
    {
        offset >>= 1u;

        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < dd)
        {
            uint ai_d = offset * ((group_thread_id << 1u) + 1u) - 1u;
            uint bi_d = offset * ((group_thread_id << 1u) + 2u) - 1u;
            ai_d = SHARED_MEMORY_ADDRESS(ai_d);
            bi_d = SHARED_MEMORY_ADDRESS(bi_d);

            DATA_TYPE temp = s_scan[ai_d];
            s_scan[ai_d] = s_scan[bi_d];
            s_scan[bi_d] += temp;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (is_inclusive_scan)
    {
        ai = group_thread_id + 1u;
        bi = ai + num_group_threads;
        ai = SHARED_MEMORY_ADDRESS(ai);
        bi = SHARED_MEMORY_ADDRESS(bi);
    }

    // copy scanned data to global memory
    if (global_ai < num_elements)
        data_buffer[global_ai] = s_scan[ai];
    if (global_bi < num_elements)
        data_buffer[global_bi] = s_scan[bi];
}

// add each group's total sum to its scan output
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void AddGroupSum(uint group_thread_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;

    DATA_TYPE group_sum = group_sum_buffer[group_id];

    uint global_ai = group_thread_id + num_elements_per_group * group_id;
    uint global_bi = global_ai + num_group_threads;

    if (global_ai < num_elements)
        data_buffer[global_ai] += group_sum;
    if (global_bi < num_elements)
        data_buffer[global_bi] += group_sum;
}


#endif /* CS_GPU_PREFIX_SCAN_HLSL */