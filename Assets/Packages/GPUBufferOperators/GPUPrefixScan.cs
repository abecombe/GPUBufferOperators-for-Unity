using System;
using System.Collections.Generic;
using UnityEngine;

namespace Abecombe.GPUBufferOperators
{
    public class GPUPrefixScan : IDisposable
    {
        public enum ScanType
        {
            Inclusive = 0,
            Exclusive
        }

        private const int MaxDispatchSize = 65535;

        protected ComputeShader PrefixScanCs;
        private int _kernelPrefixScan;
        private int _kernelAddGroupSum;

        // buffers to store the sum of values within local groups
        // size: number of groups
        private List<GraphicsBuffer> _groupSumBufferList;
        // buffer to store the total sum of values
        // size: 1
        private GraphicsBuffer _totalSumBuffer;

        private uint _totalSum = 0;
        private uint[] _totalSumArr = new uint[1];

        private bool _inited = false;

        protected virtual void LoadComputeShader()
        {
            PrefixScanCs = Resources.Load<ComputeShader>("PrefixScanCS");
        }

        private void Init()
        {
            if (!PrefixScanCs) LoadComputeShader();
            _kernelPrefixScan = PrefixScanCs.FindKernel("PrefixScan");
            _kernelAddGroupSum = PrefixScanCs.FindKernel("AddGroupSum");

            _inited = true;
        }

        // Implementation of Article "Chapter 39. Parallel Prefix Sum (Scan) with CUDA"
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // you can choose from the data types uint, int, or float.

        /// <summary>
        /// Prefix scan on dataBuffer
        /// </summary>
        /// <param name="scanType">inclusive or exclusive scan</param>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer)
        {
            Scan(scanType, dataBuffer, null, 0, false, 0);
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, false, 0);
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, false, 0);
        }

        /// <summary>
        /// Prefix scan on dataBuffer
        /// </summary>
        /// <param name="scanType">inclusive or exclusive scan</param>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(scanType, dataBuffer, null, 0, true, 0);
            totalSum = _totalSum;
        }
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer, out int totalSum)
        {
            Scan(scanType, dataBuffer, null, 0, true, 0);
            totalSum = unchecked((int)_totalSum);
        }
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer, out float totalSum)
        {
            Scan(scanType, dataBuffer, null, 0, true, 0);
            totalSum = BitConverter.ToSingle(BitConverter.GetBytes(_totalSum), 0);
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, true, 0);
            totalSum = _totalSum;
        }
        public void InclusiveScan(GraphicsBuffer dataBuffer, out int totalSum)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, true, 0);
            totalSum = unchecked((int)_totalSum);
        }
        public void InclusiveScan(GraphicsBuffer dataBuffer, out float totalSum)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, true, 0);
            totalSum = BitConverter.ToSingle(BitConverter.GetBytes(_totalSum), 0);
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, true, 0);
            totalSum = _totalSum;
        }
        public void ExclusiveScan(GraphicsBuffer dataBuffer, out int totalSum)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, true, 0);
            totalSum = unchecked((int)_totalSum);
        }
        public void ExclusiveScan(GraphicsBuffer dataBuffer, out float totalSum)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, true, 0);
            totalSum = BitConverter.ToSingle(BitConverter.GetBytes(_totalSum), 0);
        }


        /// <summary>
        /// Prefix scan on dataBuffer
        /// </summary>
        /// <param name="scanType">inclusive or exclusive scan</param>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSumBuffer">data buffer to store the total sum</param>
        /// <param name="bufferOffset">index of the element in the totalSumBuffer to store the total sum</param>
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset = 0)
        {
            Scan(scanType, dataBuffer, totalSumBuffer, bufferOffset, false, 0);
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSumBuffer">data buffer to store the total sum</param>
        /// <param name="bufferOffset">index of the element in the totalSumBuffer to store the total sum</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset = 0)
        {
            Scan(ScanType.Inclusive, dataBuffer, totalSumBuffer, bufferOffset, false, 0);
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSumBuffer">data buffer to store the total sum</param>
        /// <param name="bufferOffset">index of the element in the totalSumBuffer to store the total sum</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset = 0)
        {
            Scan(ScanType.Exclusive, dataBuffer, totalSumBuffer, bufferOffset, false, 0);
        }

        private void Scan(ScanType scanType, GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset, bool returnTotalSum, int recursiveDepth)
        {
            if (!_inited) Init();

            _totalSumBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            totalSumBuffer ??= _totalSumBuffer;

            var cs = PrefixScanCs;
            var k_scan = _kernelPrefixScan;
            var k_add = _kernelAddGroupSum;

            int numElements = dataBuffer.count;

            int numGroupThreads = SetNumGroupThreads(cs, numElements);
            int numElementsPerGroup = 2 * numGroupThreads;

            int numGroups = (numElements + numElementsPerGroup - 1) / numElementsPerGroup;

            _groupSumBufferList ??= new List<GraphicsBuffer>();
            GraphicsBuffer groupSumBuffer;
            if (_groupSumBufferList.Count == recursiveDepth)
            {
                groupSumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numGroups, sizeof(uint));
                _groupSumBufferList.Add(groupSumBuffer);
            }
            else if (_groupSumBufferList.Count > recursiveDepth)
            {
                groupSumBuffer = _groupSumBufferList[recursiveDepth];
                if (groupSumBuffer.count != numGroups)
                {
                    groupSumBuffer.Release();
                    groupSumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numGroups, sizeof(uint));
                    _groupSumBufferList[recursiveDepth] = groupSumBuffer;
                }
            }
            else
            {
                Debug.LogError("Fatal Error in Prefix Scan");
                return;
            }

            // scan input data locally and output total sums within groups
            cs.SetInt("num_elements", numElements);
            cs.SetInt("is_inclusive_scan", scanType == ScanType.Inclusive && recursiveDepth == 0 ? 1 : 0);
            cs.SetBuffer(k_scan, "data_buffer", dataBuffer);
            cs.SetBuffer(k_scan, "group_sum_buffer", groupSumBuffer);
            cs.SetInt("group_sum_offset", 0);
            for (int i = 0; i < numGroups; i += MaxDispatchSize)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_scan, Mathf.Min(numGroups - i, MaxDispatchSize), 1, 1);
            }

            // scan group total sums
            if (numGroups <= numElementsPerGroup)
            {
                cs.SetInt("num_elements", numGroups);
                cs.SetInt("is_inclusive_scan", 0);
                cs.SetInt("group_offset", 0);
                cs.SetBuffer(k_scan, "data_buffer", groupSumBuffer);
                cs.SetBuffer(k_scan, "group_sum_buffer", totalSumBuffer);
                cs.SetInt("group_sum_offset", (int)bufferOffset);
                cs.Dispatch(k_scan, 1, 1, 1);

                if (returnTotalSum)
                {
                    totalSumBuffer.GetData(_totalSumArr, 0, (int)bufferOffset, 1);
                    _totalSum = _totalSumArr[0];
                }
            }
            // execute this function recursively
            else
            {
                Scan(scanType, groupSumBuffer, totalSumBuffer, bufferOffset, returnTotalSum, recursiveDepth + 1);
            }

            // add each group's total sum to its scan output
            SetNumGroupThreads(cs, numElements);
            cs.SetInt("num_elements", numElements);
            cs.SetBuffer(k_add, "data_buffer", dataBuffer);
            cs.SetBuffer(k_add, "group_sum_buffer", groupSumBuffer);
            for (int i = 0; i < numGroups; i += MaxDispatchSize)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_add, Mathf.Min(numGroups - i, MaxDispatchSize), 1, 1);
            }
        }

        // changing the number of group threads according to the number of data to reduce the number of nests
        private static int SetNumGroupThreads(ComputeShader cs, int numElements)
        {
            switch (numElements)
            {
                case <= 65536:
                    cs.EnableKeyword("NUM_GROUP_THREADS_128");
                    cs.DisableKeyword("NUM_GROUP_THREADS_256");
                    cs.DisableKeyword("NUM_GROUP_THREADS_512");
                    return 128;
                case <= 262144:
                    cs.DisableKeyword("NUM_GROUP_THREADS_128");
                    cs.EnableKeyword("NUM_GROUP_THREADS_256");
                    cs.DisableKeyword("NUM_GROUP_THREADS_512");
                    return 256;
                default:
                    cs.DisableKeyword("NUM_GROUP_THREADS_128");
                    cs.DisableKeyword("NUM_GROUP_THREADS_256");
                    cs.EnableKeyword("NUM_GROUP_THREADS_512");
                    return 512;
            }
        }

        /// <summary>
        /// Release buffers
        /// </summary>
        public void Dispose()
        {
            if (_groupSumBufferList is not null) { _groupSumBufferList.ForEach(x => x.Release()); _groupSumBufferList = null; }
            if (_totalSumBuffer is not null) { _totalSumBuffer.Release(); _totalSumBuffer = null; }
        }
    }
}