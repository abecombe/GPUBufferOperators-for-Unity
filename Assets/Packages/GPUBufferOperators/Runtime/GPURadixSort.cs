using System;
using UnityEngine;

namespace Abecombe.GPUBufferOperators
{
    public class GPURadixSort : IDisposable
    {
        public enum KeyType
        {
            UInt = 0,
            Int,
            Float
        }

        public enum SortingOrder
        {
            Ascending = 0,
            Descending
        }

        // we use 16-way radix sort
        private const int NWay = 16;

        private const int NumGroupThreads = 128;
        private const int NumElementsPerGroup = NumGroupThreads;

        private const int MaxDispatchSize = 65535;

        protected ComputeShader RadixSortCs;
        private int _kernelComputeDispatchSize;
        private int _kernelRadixSortLocal;
        private int _kernelGlobalShuffle;

        private GPUPrefixScan _prefixScan = new();

        // buffer to store the locally sorted input data
        // size: number of data
        private GraphicsBuffer _tempBuffer;
        // buffer to store the first index of each 4bit key-value (0, 1, 2, ..., 16) within locally sorted groups
        // size: 16 * number of groups
        private GraphicsBuffer _firstIndexBuffer;
        // buffer to store the sums of each 4bit key-value (0, 1, 2, ..., 16) within locally sorted groups
        // size: 16 * number of groups
        private GraphicsBuffer _groupSumBuffer;
        // buffer to store the group size for dispatching
        // size: 4 (groupSize.x, groupSize.y, groupSize.z, groupCount)
        private GraphicsBuffer _groupSizeBuffer;

        private int[] _sortStartEndIndexArray = new int[2];
        private int[] _groupSizeArray = new int[4];

        private bool _inited = false;

        protected virtual void LoadComputeShader()
        {
            RadixSortCs = Resources.Load<ComputeShader>("GPURadixSort");
        }

        private void Init()
        {
            if (!RadixSortCs) LoadComputeShader();
            _kernelComputeDispatchSize = RadixSortCs.FindKernel("ComputeDispatchSize");
            _kernelRadixSortLocal = RadixSortCs.FindKernel("RadixSortLocal");
            _kernelGlobalShuffle = RadixSortCs.FindKernel("GlobalShuffle");
            _groupSizeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 4, sizeof(uint));

            _inited = true;
        }

        // Implementation of Paper "Fast 4-way parallel radix sorting on GPUs"
        // https://vgc.poly.edu/~csilva/papers/cgf.pdf
        // (we use 16-way radix sort)

        /// <summary>
        /// Sort data buffer in ascending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, 0, dataBuffer.count, SortingOrder.Ascending, keyType, maxValue);
        }
        /// <summary>
        /// Sort data buffer in ascending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="startIndex">start index of the data buffer for sorting</param>
        /// <param name="endIndex">end index of the data buffer for sorting</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, int startIndex, int endIndex, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, startIndex, endIndex, SortingOrder.Ascending, keyType, maxValue);
        }
        /// <summary>
        /// Sort data buffer in ascending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="sortStartEndIndexBuffer">buffer to store the [start, end) index for sorting (buffer size: 2)</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, GraphicsBuffer sortStartEndIndexBuffer, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, sortStartEndIndexBuffer, SortingOrder.Ascending, keyType, maxValue);
        }

        /// <summary>
        /// Sort data buffer in descending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void SortDescending(GraphicsBuffer dataBuffer, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, 0, dataBuffer.count, SortingOrder.Descending, keyType, maxValue);
        }
        /// <summary>
        /// Sort data buffer in descending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="startIndex">start index of the data buffer for sorting</param>
        /// <param name="endIndex">end index of the data buffer for sorting</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void SortDescending(GraphicsBuffer dataBuffer, int startIndex, int endIndex, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, startIndex, endIndex, SortingOrder.Descending, keyType, maxValue);
        }
        /// <summary>
        /// Sort data buffer in descending order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="sortStartEndIndexBuffer">buffer to store the [start, end) index for sorting (buffer size: 2)</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void SortDescending(GraphicsBuffer dataBuffer, GraphicsBuffer sortStartEndIndexBuffer, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, sortStartEndIndexBuffer, SortingOrder.Descending, keyType, maxValue);
        }

        /// <summary>
        /// Sort data buffer in specified order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="sortingOrder"> sorting order (ascending or descending)</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, SortingOrder sortingOrder, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            Sort(dataBuffer, 0, dataBuffer.count, sortingOrder, keyType, maxValue);
        }
        /// <summary>
        /// Sort data buffer in specified order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="startIndex">start index of the data buffer for sorting</param>
        /// <param name="endIndex">end index of the data buffer for sorting</param>
        /// <param name="sortingOrder"> sorting order (ascending or descending)</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, int startIndex, int endIndex, SortingOrder sortingOrder, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            if (!_inited) Init();

            var cs = RadixSortCs;
            var k_local = _kernelRadixSortLocal;
            var k_shuffle = _kernelGlobalShuffle;

            int bufferSize = dataBuffer.count;
            int maxNumGroups = (bufferSize + NumElementsPerGroup - 1) / NumElementsPerGroup;

            CheckBufferSizeChanged(bufferSize, maxNumGroups, dataBuffer.stride);

            cs.DisableKeyword("USE_INDIRECT_DISPATCH");

            cs.SetInt("key_type", (int)keyType);
            cs.SetInt("sorting_order", (int)sortingOrder);

            _sortStartEndIndexArray[0] = startIndex;
            _sortStartEndIndexArray[1] = endIndex;
            cs.SetInts("start_end_index", _sortStartEndIndexArray);

            int threadCount = endIndex - startIndex;
            int groupCount = (threadCount + NumElementsPerGroup - 1) / NumElementsPerGroup;
            Vector3Int groupSize = groupCount switch
            {
                <= MaxDispatchSize => new Vector3Int(groupCount, 1, 1),
                <= 16 * MaxDispatchSize => new Vector3Int(16, (groupCount + 15) / 16, 1),
                <= 128 * MaxDispatchSize => new Vector3Int(128, (groupCount + 127) / 128, 1),
                _ => new Vector3Int(1024, (groupCount + 1023) / 1024, 1)
            };
            _groupSizeArray[0] = groupSize.x;
            _groupSizeArray[1] = groupSize.y;
            _groupSizeArray[2] = groupSize.z;
            _groupSizeArray[3] = groupCount;
            cs.SetInts("group_size", _groupSizeArray);

            cs.SetBuffer(k_local, "data_in_buffer", dataBuffer);
            cs.SetBuffer(k_local, "data_out_buffer", _tempBuffer);
            cs.SetBuffer(k_local, "first_index_buffer", _firstIndexBuffer);
            cs.SetBuffer(k_local, "group_sum_buffer", _groupSumBuffer);

            cs.SetBuffer(k_shuffle, "data_in_buffer", _tempBuffer);
            cs.SetBuffer(k_shuffle, "data_out_buffer", dataBuffer);
            cs.SetBuffer(k_shuffle, "first_index_buffer", _firstIndexBuffer);
            cs.SetBuffer(k_shuffle, "global_prefix_sum_buffer", _groupSumBuffer);

            int firstBitHigh = keyType == KeyType.UInt ? GetHighestBitPosition(maxValue) : 32;
            for (int bitShift = 0; bitShift < firstBitHigh; bitShift += GetHighestBitPosition(NWay) - 1)
            {
                cs.SetInt("bit_shift", bitShift);

                // sort input data locally and output first-index / sums of each 4bit key-value within groups
                cs.Dispatch(k_local, groupSize.x, groupSize.y, groupSize.z);

                // prefix scan global group sum data
                _prefixScan.ExclusiveScan(_groupSumBuffer);

                // copy input data to final position in global memory
                cs.Dispatch(k_shuffle, groupSize.x, groupSize.y, groupSize.z);
            }
        }
        /// <summary>
        /// Sort data buffer in specified order
        /// </summary>
        /// <param name="dataBuffer">data buffer to be sorted</param>
        /// <param name="sortStartEndIndexBuffer">buffer to store the [start, end) index for sorting (buffer size: 2)</param>
        /// <param name="sortingOrder"> sorting order (ascending or descending)</param>
        /// <param name="keyType">sorting key type (uint, int or float)</param>
        /// <param name="maxValue">maximum key-value (valid only when keyType is UInt)</param>
        public void Sort(GraphicsBuffer dataBuffer, GraphicsBuffer sortStartEndIndexBuffer, SortingOrder sortingOrder, KeyType keyType, uint maxValue = uint.MaxValue)
        {
            if (!_inited) Init();

            var cs = RadixSortCs;
            var k_dispatch = _kernelComputeDispatchSize;
            var k_local = _kernelRadixSortLocal;
            var k_shuffle = _kernelGlobalShuffle;

            int bufferSize = dataBuffer.count;
            int maxNumGroups = (bufferSize + NumElementsPerGroup - 1) / NumElementsPerGroup;

            CheckBufferSizeChanged(bufferSize, maxNumGroups, dataBuffer.stride);

            cs.EnableKeyword("USE_INDIRECT_DISPATCH");

            cs.SetInt("key_type", (int)keyType);
            cs.SetInt("sorting_order", (int)sortingOrder);

            cs.SetBuffer(k_dispatch, "start_end_index_buffer", sortStartEndIndexBuffer);
            cs.SetBuffer(k_dispatch, "group_size_buffer_write", _groupSizeBuffer);
            cs.Dispatch(k_dispatch, 1, 1, 1);

            cs.SetBuffer(k_local, "data_in_buffer", dataBuffer);
            cs.SetBuffer(k_local, "data_out_buffer", _tempBuffer);
            cs.SetBuffer(k_local, "start_end_index_buffer", sortStartEndIndexBuffer);
            cs.SetBuffer(k_local, "first_index_buffer", _firstIndexBuffer);
            cs.SetBuffer(k_local, "group_sum_buffer", _groupSumBuffer);
            cs.SetBuffer(k_local, "group_size_buffer_read", _groupSizeBuffer);

            cs.SetBuffer(k_shuffle, "data_in_buffer", _tempBuffer);
            cs.SetBuffer(k_shuffle, "data_out_buffer", dataBuffer);
            cs.SetBuffer(k_shuffle, "start_end_index_buffer", sortStartEndIndexBuffer);
            cs.SetBuffer(k_shuffle, "first_index_buffer", _firstIndexBuffer);
            cs.SetBuffer(k_shuffle, "global_prefix_sum_buffer", _groupSumBuffer);
            cs.SetBuffer(k_shuffle, "group_size_buffer_read", _groupSizeBuffer);

            int firstBitHigh = keyType == KeyType.UInt ? GetHighestBitPosition(maxValue) : 32;
            for (int bitShift = 0; bitShift < firstBitHigh; bitShift += GetHighestBitPosition(NWay) - 1)
            {
                cs.SetInt("bit_shift", bitShift);

                // sort input data locally and output first-index / sums of each 4bit key-value within groups
                cs.DispatchIndirect(k_local, _groupSizeBuffer);

                // prefix scan global group sum data
                _prefixScan.ExclusiveScan(_groupSumBuffer);

                // copy input data to final position in global memory
                cs.DispatchIndirect(k_shuffle, _groupSizeBuffer);
            }
        }

        private void CheckBufferSizeChanged(int bufferSize, int maxNumGroups, int bufferStride)
        {
            if (_tempBuffer is null || _tempBuffer.count < bufferSize || _tempBuffer.stride != bufferStride)
            {
                _tempBuffer?.Release();
                _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, bufferStride);
            }
            if (_firstIndexBuffer is null || _firstIndexBuffer.count < NWay * maxNumGroups)
            {
                _firstIndexBuffer?.Release();
                _firstIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, NWay * maxNumGroups, sizeof(uint));
            }
            if (_groupSumBuffer is null || _groupSumBuffer.count < NWay * maxNumGroups)
            {
                _groupSumBuffer?.Release();
                _groupSumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, NWay * maxNumGroups, sizeof(uint));
            }
        }

        private static int GetHighestBitPosition(uint value)
        {
            int position = 0;
            while (value != 0)
            {
                value >>= 1;
                position++;
            }

            return position;
        }

        /// <summary>
        /// Release buffers
        /// </summary>
        public void Dispose()
        {
            if (!_inited) return;

            if (_tempBuffer is not null) { _tempBuffer.Release(); _tempBuffer = null; }
            if (_firstIndexBuffer is not null) { _firstIndexBuffer.Release(); _firstIndexBuffer = null; }
            if (_groupSumBuffer is not null) { _groupSumBuffer.Release(); _groupSumBuffer = null; }
            if (_groupSizeBuffer is not null) { _groupSizeBuffer.Release(); _groupSizeBuffer = null; }

            _prefixScan.Dispose();

            _inited = false;
        }
    }
}