using System.Linq;
using Abecombe.GPUBufferOperators;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

using Random = UnityEngine.Random;

public class ShufflingSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private int _shufflingKey = 0;

    private GPUShuffling _shuffling = new();

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

        uint2[] dataArr1 = new uint2[_numData];
        uint2[] dataArr2 = new uint2[_numData];

        Random.InitState(_shufflingKey);
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, 1000000);
            dataArr1[i] = new uint2(value, i);
        }
        _tempBuffer.SetData(dataArr1);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopyShufflingBuffer");

        _copyCs.SetBuffer(_copyKernel, "shuffling_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "shuffling_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);

        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _shuffling.Shuffle(_dataBuffer, _shufflingKey);

        _dataBuffer.GetData(dataArr2);
        dataArr2 = dataArr2.OrderBy(data => data.y).ToArray();
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr1[i].x != dataArr2[i].x || dataArr1[i].y != dataArr2[i].y)
            {
                Debug.LogError("Shuffling Failure");
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

        _shuffling.Shuffle(_dataBuffer, _shufflingKey);
    }

    private void OnDestroy()
    {
        _shuffling.Dispose();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}