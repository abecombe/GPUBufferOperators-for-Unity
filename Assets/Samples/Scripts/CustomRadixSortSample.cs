using System.Collections.Generic;
using System.Linq;
using Abecombe.GPUBufferOperators;
using System.Runtime.InteropServices;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

using Random = UnityEngine.Random;

public struct CustomStruct
{
    public float Key { get; }
    public uint ID { get; }
    public float Dummy1 { get; }
    public float Dummy2 { get; }

    public CustomStruct(float key, uint id, float dummy1, float dummy2)
    {
        Key = key;
        ID = id;
        Dummy1 = dummy1;
        Dummy2 = dummy2;
    }
}

public class CustomRadixSort : GPURadixSort
{
    protected override void LoadComputeShader()
    {
        RadixSortCs = Resources.Load<ComputeShader>("CustomRadixSortCS");
    }
}

public class CustomRadixSortSample : MonoBehaviour
{
    [SerializeField] private int _numData = 100;
    [SerializeField] private int _randomSeed = 0;

    private CustomRadixSort _radixSort = new();

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private ComputeShader _copyCs;
    private int _copyKernel;

    private const int NumGroupThreads = 128;
    private const int MaxDispatchSize = 65535;
    private int DispatchSize => (_numData + NumGroupThreads - 1) / NumGroupThreads;

    private void Start()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(CustomStruct)));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(CustomStruct)));

        CustomStruct[] dataArr = new CustomStruct[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            float value = Random.Range(-10000f, 10000f);
            dataArr[i] = new CustomStruct(value, i, 10f, 20f);
        }
        _tempBuffer.SetData(dataArr);

        _copyCs = Resources.Load<ComputeShader>("CopyCS");
        _copyKernel = _copyCs.FindKernel("CopyCustomSortBuffer");

        _copyCs.SetBuffer(_copyKernel, "custom_sort_data_buffer", _dataBuffer);
        _copyCs.SetBuffer(_copyKernel, "custom_sort_temp_buffer", _tempBuffer);
        _copyCs.SetInt("num_elements", _numData);
    }

    private void Update()
    {
        for (int i = 0; i < DispatchSize; i += MaxDispatchSize)
        {
            _copyCs.SetInt("group_offset", i);
            _copyCs.Dispatch(_copyKernel, Mathf.Min(DispatchSize - i, MaxDispatchSize), 1, 1);
        }

        _radixSort.SortDescending(_dataBuffer, GPURadixSort.KeyType.Float);
    }

    private void OnDestroy()
    {
        _radixSort.Dispose();

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

        CustomStruct[] dataArr1 = new CustomStruct[_numData];
        _dataBuffer.GetData(dataArr1);

        _radixSort.SortDescending(_dataBuffer, GPURadixSort.KeyType.Float);

        dataArr1 = dataArr1.OrderByDescending(data => data.Key).ToArray();

        CustomStruct[] dataArr2 = new CustomStruct[_numData];
        _dataBuffer.GetData(dataArr2);

        if (dataArr1.SequenceEqual(dataArr2))
        {
            Debug.Log("Sorting Success");
        }
        else
        {
            Debug.LogError("Sorting Failure");
        }

        OnDestroy();
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(CustomRadixSortSample))]
public class CustomRadixSortSampleEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        GUILayout.Space(5f);
        if (GUILayout.Button("Check Success"))
        {
            var customRadixSortSample = target as CustomRadixSortSample;
            customRadixSortSample.CheckSuccess();
        }
    }
}
#endif