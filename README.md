# GPU Radix Sort for Unity

**GPU Radix Sort using Compute Shader**

**Based on [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/publications/Ha2009b/Ha_CGF2009.pdf)**

**The key type used for sorting is limited to `uint`, `int` or `float`.**

**No restrictions on input data struct or size.**

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
**`uint2` is an example of a data struct & you can change it.**  
**Note that the larger the data struct size, the longer it takes to sort.**

### Sort
```csharp
radixSort.Sort(GraphicsBuffer DataBuffer, SortingOrder SortingOrder, KeyType KeyType, uint MaxValue = uint.MaxValue);
```
* **DataBuffer**
  * data buffer to be sorted

* **SortingOrder**
  * sorting order (ascending or descending)

* **KeyType**
  * sorting key type (uint, int or float)

* **MaxValue**
  * maximum key-value (valid only when KeyType is UInt)
  * **since this variable directly related to the algorithmic complexity, passing this argument will reduce the cost of sorting.**

### Dispose
```csharp
void OnDestroy() {
  radixSort.Dispose();
}
```

# GPU Prefix Scan for Unity

**GPU Exclusive Prefix Scan using Compute Shader**

**Based on [Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)**

**The data struct is limited to `uint`, `int` or `float`.**

**No restrictions on input data size.**

## Usage
### Init
***C# code***
```csharp
GPUPrefixScan prefixScan = new();
```
***PrefixScanCS.compute***
```text
#define DATA_TYPE uint
// you can choose from the data types uint, int, or float.
```

### Scan
```csharp
prefixScan.Scan(GraphicsBuffer DataBuffer, out uint TotalSum);
```
* **DataBuffer**
  * data buffer to be scanned

* **TotalSum**
  * the total sum of values

### Dispose
```csharp
void OnDestroy() {
  prefixScan.Dispose();
}
```

# GPU Shuffling for Unity

**GPU Shuffling using Compute Shader**

**Based on [Bandwidth-Optimal Random Shuffling for GPUs](https://arxiv.org/pdf/2106.06161)**

**No restrictions on input data struct or size.**

## Usage
### Init
***C# code***
```csharp
GPUShuffling shuffling = new();
```
***ShufflingCS.compute***
```text
#define DATA_TYPE uint2  // input data struct
```
**`uint2` is an example of a data struct & you can change it.**  
**Note that the larger the data struct size, the longer it takes to shuffle.**

### Shuffle
```csharp
shuffling.Shuffle(GraphicsBuffer DataBuffer, int Key);
```
* **DataBuffer**
  * data buffer to be shuffled

* **Key**
  * key for shuffling

### Dispose
```csharp
void OnDestroy() {
  shuffling.Dispose();
}
```

# GPU Filtering for Unity

**GPU Filtering using Compute Shader**

**Filtering: Gather elements that meet certain condition to the front of the buffer**

**No restrictions on input data struct or size.**

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
**`uint2` is an example of a data struct & you can change it.**  
**Note that the larger the data struct size, the longer it takes to filter.**

### Filter
```csharp
filtering.Filter(GraphicsBuffer DataBuffer, out uint NumFilteredElements);
```
* **DataBuffer**
  * data buffer to be filtered

* **NumFilteredElements**
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
* **[Bandwidth-Optimal Random Shuffling for GPUs](https://arxiv.org/pdf/2106.06161)**  