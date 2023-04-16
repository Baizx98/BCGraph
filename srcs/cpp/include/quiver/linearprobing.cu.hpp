#pragma once
#include <stdio.h>
#include <stdint.h>
#include <vector>
#define BIG_CONSTANT(x) (x)
//#define BIG_CONSTANT(x) (x##LLU)
struct KeyValue
{
    unsigned long long key;
    uint64_t value;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;
const uint64_t kNumKeyValues = kHashTableCapacity / 2;
const unsigned long long kEmpty = 0xffffffffffffffff;

KeyValue* create_hashtable();
void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint64_t num_kvs);
void partern_erase_hashtable(KeyValue*pHashTable,uint64_t erase_size);
std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);


KeyValue* create_hashtable() 
{
    // Allocate memory
    KeyValue* hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffffffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}
__device__ uint64_t hash( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k & (kHashTableCapacity-1);
}
// Insert the key/values in kvs into the hashtable
__global__ void fgpu_hashtable_insert(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        unsigned long long key = kvs[threadid].key;
        uint64_t value = kvs[threadid].value;
        uint64_t slot = hash(key);

        while (true)
        {
            unsigned long long prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                hashtable[slot].value = value;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity-1);    //开发地址法：线性探查
        }
    }
}
 
void insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, fgpu_hashtable_insert, 0, 0);

    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    fgpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaFree(device_kvs);
}
__global__ void gpu_perase_hashtable(KeyValue* pHashTable,uint64_t* dsize)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity) 
    {
        if (pHashTable[threadid].key != kEmpty) 
        {
            uint64_t value = pHashTable[threadid].value;
            if (value != kEmpty && value>=*dsize)
            {
                pHashTable[threadid].key=kEmpty;
                pHashTable[threadid].value=kEmpty;
            }
        }
    }
}
void partern_erase_hashtable(KeyValue*pHashTable,uint64_t erase_size)
{
    uint64_t*dsize;
    cudaMalloc(&dsize,sizeof(uint64_t));
    cudaMemcpy(dsize,&erase_size,sizeof(uint64_t),cudaMemcpyHostToDevice);
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_perase_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_perase_hashtable<<<gridsize, threadblocksize>>>(pHashTable,dsize);

    cudaFree(dsize);

}
__global__ void gpu_iterate_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t* kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity) 
    {
        if (pHashTable[threadid].key != kEmpty) 
        {
            uint64_t value = pHashTable[threadid].value;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    uint32_t* device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);


    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
    cudaFree(device_num_kvs);
    return kvs;
}

