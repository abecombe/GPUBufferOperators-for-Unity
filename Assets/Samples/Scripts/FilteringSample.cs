using Abecombe.GPUBufferOperators;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

using Random = UnityEngine.Random;

public class FilteringSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private int _randomSeed = 0;

    private GPUFiltering _filtering = new();

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
        uint2[] filterDataArr = new uint2[_numData];

        Random.InitState(_randomSeed);
        uint sum1 = 0;
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, 2);
            dataArr[i] = new uint2(value, i);
            if (value == 1)
                filterDataArr[sum1++] = new uint2(value, i);
        }
        _tempBuffer.SetData(dataArr);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopyFilteringBuffer");

        _copyCs.SetBuffer(_copyKernel, "filtering_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "filtering_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);

        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        uint sum2 = _filtering.Filter(_dataBuffer, true);

        if (sum1 != sum2)
        {
            Debug.LogError("Filtering Failure");
        }

        _dataBuffer.GetData(dataArr);
        for (int i = 0; i < sum1; i++)
        {
            if (math.any(dataArr[i] != filterDataArr[i]))
            {
                Debug.LogError("Filtering Failure");
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

        _filtering.Filter(_dataBuffer);
    }

    private void OnDestroy()
    {
        _filtering.Dispose();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}