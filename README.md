# GPU Radix Sort for Unity

**GPU Radix Sort using Compute Shader**

**Based on [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/publications/Ha2009b/Ha_CGF2009.pdf)**

**The key type used for sorting is limited to `uint`.**

**No restrictions on input data type or size.**

## Algorithmic complexity
GPURadixSort has **`O(n * s * w)`** complexity  
```text
n : number of data
s : size of data struct
w : number of bits to sort
```

## Usage
### Init
***C# code***
```csharp
GPURadixSort radixSort = new();
```
***RadixSortCS.compute***
```text
#define DATA_TYPE uint2  // input data struct
#define GET_KEY(s) s.x   // how to get the key-values used for sorting
```
**`uint2` is an example of a data type & you can change it.**  
**Note that the larger the data struct size, the longer it takes to sort.**

### Sort
```csharp
radixSort.Sort(GraphicsBuffer DataBuffer, uint MaxValue = uint.MaxValue);
```
* **DataBuffer**  
  * data buffer to be sorted

* **MaxValue**  
  * maximum key-value  
  * **since this variable directly related to the algorithmic complexity, passing this argument will reduce the cost of sorting.**

### Dispose
```csharp
void OnDestroy() {
  radixSort.Dispose();
}
```

# GPU Filtering for Unity

**GPU Filtering using Compute Shader**

**Filtering: Gather elements that meet certain condition to the front of the buffer**

**No restrictions on input data type or size.**

## Usage
### Init
***C# code***
```csharp
GPUFiltering filtering = new();
```
***FilteringCS.compute***
```text
#define DATA_TYPE uint2        // input data struct
#define GET_KEY(s) (s.x == 1)  // certain condition used for filtering
```
**`uint2` is an example of a data type & you can change it.**  
**Note that the larger the data struct size, the longer it takes to filter.**

### Filter
```csharp
filtering.Filter(GraphicsBuffer DataBuffer, out uint numFilteredElements);
```
* **DataBuffer**  
  * data buffer to be filtered

* **numFilteredElements**  
  * the number of filtered elements

### Dispose
```csharp
void OnDestroy() {
  filtering.Dispose();
}
```

# References
* **[Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/publications/Ha2009b/Ha_CGF2009.pdf)**  
* **[Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)**  
* **[GPU Radix Sort](https://github.com/mark-poscablo/gpu-radix-sort)**