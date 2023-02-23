using Abecombe.GPUBufferOperators;
using UnityEngine;

using Random = UnityEngine.Random;

public class PrefixScanSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private uint _randomValueMax = 100;
    [SerializeField] private int _randomSeed = 0;

    private GPUPrefixScan _prefixScan = new();

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private ComputeShader _copyCs;
    private int _copyKernel;

    private const int NumGroupThreads = 128;
    private const int MaxDispatchSize = 65535;
    private int DispatchSize => (_numData + NumGroupThreads - 1) / NumGroupThreads;

    private void Start()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));

        uint[] dataArr1 = new uint[_numData];
        uint[] dataArr2 = new uint[_numData];

        Random.InitState(_randomSeed);
        uint sum1 = 0;
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            sum1 += value;
            dataArr1[i] = value;
        }
        _tempBuffer.SetData(dataArr1);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopyScanBuffer");

        _copyCs.SetBuffer(_copyKernel, "scan_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "scan_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);

        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        uint sum2 = _prefixScan.Scan(_dataBuffer, true);

        _dataBuffer.GetData(dataArr2);
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr2[i + 1] - dataArr2[i] != dataArr1[i])
            {
                Debug.LogError("Scanning Failure");
                break;
            }
        }

        if (sum1 != sum2)
        {
            Debug.LogError("Scanning Failure");
        }
    }

    private void Update()
    {
        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _prefixScan.Scan(_dataBuffer);
    }

    private void OnDestroy()
    {
        _prefixScan.Dispose();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}