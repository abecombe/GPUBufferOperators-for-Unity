using System;
using UnityEngine;

namespace Abecombe.GPUBufferOperators
{
    public class GPUOneSweepPrefixScan : IDisposable
    {
        public enum ScanType
        {
            Inclusive = 0,
            Exclusive
        }

        private const int NumGroupThreads = 128;
        private const int NumElementsPerGroup = 2 * NumGroupThreads;

        private const int MaxDispatchSize = 65535;

        protected ComputeShader PrefixScanCs;
        private int _kernelClearBuffer;
        private int _kernelPrefixScan;

        // buffer for calculating the partition index of each group
        // size: 1
        private GraphicsBuffer _partitionIndexBuffer;
        // buffer for storing the partition descriptor of each group
        // size: number of groups
        private GraphicsBuffer _partitionDescriptorBuffer;

        private uint _totalSum = 0;
        private uint[] _totalSumArr = new uint[1];

        private bool _inited = false;

        protected virtual void LoadComputeShader()
        {
            PrefixScanCs = Resources.Load<ComputeShader>("GPUOneSweepPrefixScan");
        }

        private void Init()
        {
            if (!PrefixScanCs) LoadComputeShader();
            _kernelClearBuffer = PrefixScanCs.FindKernel("ClearBuffer");
            _kernelPrefixScan = PrefixScanCs.FindKernel("PrefixScan");

            _inited = true;
        }

        // Implementation of Article "Single-pass Parallel Prefix Scan with Decoupled Look-back"
        // https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
        // Only support UInt data type

        /// <summary>
        /// Prefix scan on dataBuffer
        /// </summary>
        /// <param name="scanType">inclusive or exclusive scan</param>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer)
        {
            Scan(scanType, dataBuffer, null, 0, false);
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, false);
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, false);
        }

        /// <summary>
        /// Prefix scan on dataBuffer
        /// </summary>
        /// <param name="scanType">inclusive or exclusive scan</param>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void Scan(ScanType scanType, GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(scanType, dataBuffer, null, 0, true);
            totalSum = _totalSum;
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(ScanType.Inclusive, dataBuffer, null, 0, true);
            totalSum = _totalSum;
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSum">the total sum of values</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer, out uint totalSum)
        {
            Scan(ScanType.Exclusive, dataBuffer, null, 0, true);
            totalSum = _totalSum;
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
            Scan(scanType, dataBuffer, totalSumBuffer, bufferOffset, false);
        }
        /// <summary>
        /// Inclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSumBuffer">data buffer to store the total sum</param>
        /// <param name="bufferOffset">index of the element in the totalSumBuffer to store the total sum</param>
        public void InclusiveScan(GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset = 0)
        {
            Scan(ScanType.Inclusive, dataBuffer, totalSumBuffer, bufferOffset, false);
        }
        /// <summary>
        /// Exclusive Prefix scan on dataBuffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be scanned</param>
        /// <param name="totalSumBuffer">data buffer to store the total sum</param>
        /// <param name="bufferOffset">index of the element in the totalSumBuffer to store the total sum</param>
        /// <param name="dataType">data type (uint, int or float)</param>
        public void ExclusiveScan(GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset = 0)
        {
            Scan(ScanType.Exclusive, dataBuffer, totalSumBuffer, bufferOffset, false);
        }

        private void Scan(ScanType scanType, GraphicsBuffer dataBuffer, GraphicsBuffer totalSumBuffer, uint bufferOffset, bool returnTotalSum)
        {
            if (!_inited) Init();

            var cs = PrefixScanCs;
            var k_clear = _kernelClearBuffer;
            var k_scan = _kernelPrefixScan;

            int numElements = dataBuffer.count;
            int numGroups = (numElements + NumElementsPerGroup - 1) / NumElementsPerGroup;

            _partitionIndexBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            if (_partitionDescriptorBuffer is null || _partitionDescriptorBuffer.count < numGroups)
            {
                _partitionDescriptorBuffer?.Release();
                _partitionDescriptorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numGroups, sizeof(uint));
            }

            // clear buffers
            int numGroupsForClearBuffer = (numGroups + NumGroupThreads - 1) / NumGroupThreads;
            cs.SetInt("num_elements", numGroups);
            cs.SetBuffer(k_clear, "partition_index_buffer", _partitionIndexBuffer);
            cs.SetBuffer(k_clear, "partition_descriptor_buffer", _partitionDescriptorBuffer);
            for (int i = 0; i < numGroupsForClearBuffer; i += MaxDispatchSize)
            {
                cs.SetInt("thread_offset", i * NumGroupThreads);
                cs.Dispatch(k_clear, Mathf.Min(numGroupsForClearBuffer - i, MaxDispatchSize), 1, 1);
            }

            // scan data
            cs.SetInt("num_elements", numElements);
            cs.SetInt("is_inclusive_scan", scanType == ScanType.Inclusive ? 1 : 0);
            cs.SetBuffer(k_scan, "data_buffer", dataBuffer);
            cs.SetBuffer(k_scan, "partition_index_buffer", _partitionIndexBuffer);
            cs.SetBuffer(k_scan, "partition_descriptor_buffer", _partitionDescriptorBuffer);
            for (int i = 0; i < numGroups; i += MaxDispatchSize)
            {
                cs.Dispatch(k_scan, Mathf.Min(numGroups - i, MaxDispatchSize), 1, 1);
            }
        }

        /// <summary>
        /// Release buffers
        /// </summary>
        public void Dispose()
        {
            if (_partitionIndexBuffer is not null) { _partitionIndexBuffer.Release(); _partitionIndexBuffer = null; }
            if (_partitionDescriptorBuffer is not null) { _partitionDescriptorBuffer.Release(); _partitionDescriptorBuffer = null; }
        }
    }
}