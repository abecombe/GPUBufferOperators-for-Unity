using Abecombe.GPUBufferOperators;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

using Random = UnityEngine.Random;

public class RadixSortSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private uint _randomValueMax = 100;
    [SerializeField] private int _randomSeed = 0;

    private GPURadixSort _radixSort = new();

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private ComputeShader _copyCs;
    private int _copyKernel;

    private const int NumGroupThreads = 128;
    private const int MaxDispatchSize = 65535;
    private int DispatchSize => (_numData + NumGroupThreads - 1) / NumGroupThreads;

    private void Start()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(uint2)));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(uint2)));

        uint2[] dataArr = new uint2[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            dataArr[i] = new uint2(value, i);
        }
        _tempBuffer.SetData(dataArr);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopySortBuffer");

        _copyCs.SetBuffer(_copyKernel, "sort_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "sort_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);

        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _radixSort.Sort(_dataBuffer, _randomValueMax);

        _dataBuffer.GetData(dataArr);
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr[i + 1].x < dataArr[i].x)
            {
                Debug.LogError("Sorting Failure");
                break;
            }
        }
    }

    private void Update()
    {
        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _radixSort.Sort(_dataBuffer, _randomValueMax);
    }

    private void OnDestroy()
    {
        _radixSort.Dispose();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}