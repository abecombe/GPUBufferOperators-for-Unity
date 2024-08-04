using System;
using UnityEngine;

namespace Abecombe.GPUBufferOperators
{
    public class GPUShuffling : IDisposable
    {
        private const int NumGroupThreads = 128;
        private const int NumElementsPerGroup = NumGroupThreads;

        private const int MaxDispatchSize = 65535;

        protected ComputeShader ShufflingCs;
        private int _kernelApplyBijectiveFunction;
        private int _kernelShuffleElements;
        private int _kernelCopyBuffer;

        private GPUPrefixScan _prefixScan = new();

        // buffer to store the shuffled data
        // size: number of data
        private GraphicsBuffer _tempBuffer;
        // buffer to store the flag data for prefix scan
        // size: number of power of two data
        private GraphicsBuffer _flagScanBuffer;
        // buffer to store the bijection shuffle data
        // size: number of power of two data
        private GraphicsBuffer _bijectionShuffleBuffer;

        private int[] _keys = new int[4];

        private bool _inited = false;

        protected virtual void LoadComputeShader()
        {
            ShufflingCs = Resources.Load<ComputeShader>("ShufflingCS");
        }

        private void Init()
        {
            if (!ShufflingCs) LoadComputeShader();
            _kernelApplyBijectiveFunction = ShufflingCs.FindKernel("ApplyBijectiveFunction");
            _kernelShuffleElements = ShufflingCs.FindKernel("ShuffleElements");
            _kernelCopyBuffer = ShufflingCs.FindKernel("CopyBuffer");

            _inited = true;
        }

        // Implementation of Paper "Bandwidth-Optimal Random Shuffling for GPUs"
        // https://arxiv.org/pdf/2106.06161

        /// <summary>
        /// Shuffle data buffer
        /// </summary>
        /// <param name="dataBuffer">data buffer to be shuffled</param>
        /// <param name="key">key for shuffling</param>
        /// <param name="numRounds">number of rounds for shuffling (range: 1 ~ 4)</param>
        public void Shuffle(GraphicsBuffer dataBuffer, int key, int numRounds = 2)
        {
            Shuffle(dataBuffer, null, key, numRounds);
        }

        /// <summary>
        /// Shuffle data buffer
        /// </summary>
        /// <param name="dataInBuffer">input data buffer</param>
        /// <param name="dataOutBuffer">output shuffled data buffer</param>
        /// <param name="key">key for shuffling</param>
        /// <param name="numRounds">number of rounds for shuffling (range: 1 ~ 4)</param>
        public void Shuffle(GraphicsBuffer dataInBuffer, GraphicsBuffer dataOutBuffer, int key, int numRounds = 2)
        {
            if (!_inited) Init();

            bool dataOutBufferIsNull = dataOutBuffer is null;
            if (!dataOutBufferIsNull)
            {
                if (dataOutBuffer.count != dataInBuffer.count || dataOutBuffer.stride != dataInBuffer.stride)
                {
                    Debug.LogError("dataOutBuffer size and stride must be the same as dataInBuffer");
                    return;
                }
            }

            var cs = ShufflingCs;
            var k_bijection = _kernelApplyBijectiveFunction;
            var k_shuffle = _kernelShuffleElements;
            var k_copy = _kernelCopyBuffer;

            int numElements = dataInBuffer.count;
            int numPowerOfTwoElements = Mathf.NextPowerOfTwo(numElements);
            int numGroups = (numElements + NumElementsPerGroup - 1) / NumElementsPerGroup;
            int numPowerOfTwoGroups = (numPowerOfTwoElements + NumElementsPerGroup - 1) / NumElementsPerGroup;

            CheckBufferSizeChanged(numElements, numPowerOfTwoElements, dataInBuffer.stride, dataOutBufferIsNull);

            cs.SetInt("num_elements", numElements);
            cs.SetInt("num_pow_of_2_elements", numPowerOfTwoElements);

            // apply bijective function to [0, num_pow_of_2_elements] and output the result
            int totalBits = Convert.ToString(numPowerOfTwoElements, 2).Length - 1;
            int leftSideBits = totalBits / 2;
            int leftSideMask = (1 << leftSideBits) - 1;
            int rightSideBits = totalBits - leftSideBits;
            int rightSideMask = (1 << rightSideBits) - 1;
            numRounds = Mathf.Clamp(numRounds, 1, 4);
            for (int i = 0; i < numRounds; i++)
            {
                _keys[i] = Hash(i == 0 ? key : _keys[i - 1]);
            }
            cs.SetInt("left_side_bits", leftSideBits);
            cs.SetInt("left_side_mask", leftSideMask);
            cs.SetInt("right_side_bits", rightSideBits);
            cs.SetInt("right_side_mask", rightSideMask);
            cs.SetInt("num_rounds", numRounds);
            cs.SetInts("key", _keys);
            cs.SetBuffer(k_bijection, "bijection_shuffle_buffer_write", _bijectionShuffleBuffer);
            cs.SetBuffer(k_bijection, "frag_scan_buffer_write", _flagScanBuffer);
            for (int i = 0; i < numPowerOfTwoGroups; i += MaxDispatchSize)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_bijection, Mathf.Min(numPowerOfTwoGroups - i, MaxDispatchSize), 1, 1);
            }

            // prefix scan flag data
            _prefixScan.Scan(_flagScanBuffer);

            // shuffle elements based on the bijection_shuffle_buffer
            cs.SetBuffer(k_shuffle, "data_in_buffer", dataInBuffer);
            cs.SetBuffer(k_shuffle, "data_out_buffer", dataOutBufferIsNull ? _tempBuffer : dataOutBuffer);
            cs.SetBuffer(k_shuffle, "bijection_shuffle_buffer_read", _bijectionShuffleBuffer);
            cs.SetBuffer(k_shuffle, "frag_scan_buffer_read", _flagScanBuffer);
            for (int i = 0; i < numPowerOfTwoGroups; i += MaxDispatchSize)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_shuffle, Mathf.Min(numPowerOfTwoGroups - i, MaxDispatchSize), 1, 1);
            }

            if (!dataOutBufferIsNull) return;

            // copy temp buffer to data buffer
            cs.SetBuffer(k_copy, "data_in_buffer", _tempBuffer);
            cs.SetBuffer(k_copy, "data_out_buffer", dataInBuffer);
            for (int i = 0; i < numGroups; i += MaxDispatchSize)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_copy, Mathf.Min(numGroups - i, MaxDispatchSize), 1, 1);
            }
        }

        private void CheckBufferSizeChanged(int numElements, int numPowerOfTwoElements, int bufferStride, bool dataOutBufferIsNull)
        {
            if (dataOutBufferIsNull && (_tempBuffer is null || _tempBuffer.count < numElements || _tempBuffer.stride != bufferStride))
            {
                _tempBuffer?.Release();
                _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numElements, bufferStride);
            }
            if (_bijectionShuffleBuffer is null || _bijectionShuffleBuffer.count != numPowerOfTwoElements)
            {
                _bijectionShuffleBuffer?.Release();
                _bijectionShuffleBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numPowerOfTwoElements, sizeof(uint));
            }
            if (_flagScanBuffer is null || _flagScanBuffer.count != numPowerOfTwoElements)
            {
                _flagScanBuffer?.Release();
                _flagScanBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numPowerOfTwoElements, sizeof(uint));
            }
        }

        /// <summary>
        /// Hash Function using Permuted Congruential Generator
        /// Original source code: https://www.shadertoy.com/view/XlGcRh
        /// </summary>
        private static int Hash(int value)
        {
            uint uintVal = unchecked((uint)value);
            uintVal = uintVal * 747796405u + 2891336453u;
            uintVal = ((uintVal >> (int)((uintVal >> 28) + 4u)) ^ uintVal) * 277803737u;
            return unchecked((int)((uintVal >> 22) ^ uintVal));
        }

        /// <summary>
        /// Release buffers
        /// </summary>
        public void Dispose()
        {
            if (_tempBuffer is not null) { _tempBuffer.Release(); _tempBuffer = null; }
            if (_bijectionShuffleBuffer is not null) { _bijectionShuffleBuffer.Release(); _bijectionShuffleBuffer = null; }
            if (_flagScanBuffer is not null) { _flagScanBuffer.Release(); _flagScanBuffer = null; }

            _prefixScan.Dispose();
        }
    }
}