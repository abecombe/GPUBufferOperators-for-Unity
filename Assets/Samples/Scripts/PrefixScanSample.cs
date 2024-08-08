using System.Linq;
using Abecombe.GPUBufferOperators;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using Random = UnityEngine.Random;

public class PrefixScanSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private uint _randomValueMax = 100;
    [SerializeField] private GPUPrefixScan.ScanType _scanType = GPUPrefixScan.ScanType.Inclusive;
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

        uint[] dataArr = new uint[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            dataArr[i] = value;
        }
        _tempBuffer.SetData(dataArr);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopyScanBuffer");

        _copyCs.SetBuffer(_copyKernel, "scan_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "scan_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);
    }

    private void Update()
    {
        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _prefixScan.Scan(_scanType, _dataBuffer);
    }

    private void OnDestroy()
    {
        _prefixScan.Dispose();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }

    public void CheckSuccess()
    {
        Start();

        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        uint[] dataArr1 = new uint[_numData];
        _dataBuffer.GetData(dataArr1);

        uint[] scanDataArr = new uint[_numData];

        uint sum1 = 0;
        for (uint i = 0; i < _numData; i++)
        {
            switch (_scanType)
            {
                case GPUPrefixScan.ScanType.Inclusive:
                    sum1 += dataArr1[i];
                    scanDataArr[i] = sum1;
                    break;
                case GPUPrefixScan.ScanType.Exclusive:
                default:
                    scanDataArr[i] = sum1;
                    sum1 += dataArr1[i];
                    break;
            }
        }

        _prefixScan.Scan(_scanType, _dataBuffer, out uint sum2);

        uint[] dataArr2 = new uint[_numData];
        _dataBuffer.GetData(dataArr2);

        if (sum1 != sum2)
        {
            Debug.LogError("Scanning Failure");
        }
        else
        {
            if (scanDataArr.SequenceEqual(dataArr2))
            {
                Debug.Log("Scanning Success");
            }
            else
            {
                Debug.LogError("Scanning Failure");
            }
        }

        OnDestroy();
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(PrefixScanSample))]
public class PrefixScanSampleEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        GUILayout.Space(5f);
        if (GUILayout.Button("Check Success"))
        {
            var prefixScanSample = target as PrefixScanSample;
            prefixScanSample.CheckSuccess();
        }
    }
}
#endif