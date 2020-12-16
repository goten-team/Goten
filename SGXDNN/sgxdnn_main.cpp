#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#include <malloc.h>
#else
#include "Enclave.h"
#include "sgx_tseal.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#endif

#include "sgxdnn_main.hpp"
#include "randpool.hpp"
#include "utils.hpp"

#include "common_with_enclaves.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <cstring>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <cstdlib>
#include <mutex>
#include <stack>

#include "Crypto.h"

#include "../App/common_utils.cpp"

using namespace std;

using std::shared_ptr;
using std::make_shared;
using std::unordered_map;
using std::string;
using defer = shared_ptr<void>;


//using namespace SGXDNN;

int p_int = PrimeLimit;
float p = (float) p_int;
float mid = (float) (p_int / 2);

// some vectorized constants
__m256 p8f = _mm256_set1_ps(p);
__m256 p28f = _mm256_set1_ps(p * 2);
__m256 mid8f = _mm256_set1_ps(mid);
__m256 pmid8f = _mm256_set1_ps(p + mid);
__m256 negmid8f = _mm256_set1_ps(-mid - 1);
__m256 zero8f = _mm256_set1_ps((float)(0));
__m256 inv_shift8f = _mm256_set1_ps((float)(1.0/256));
__m256 six8f = _mm256_set1_ps((float) 6 * 256 * 256);

inline void MoveDown(float* input, float* out, int num_elements) {
	for(size_t i = 0; i < num_elements; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &input[i] );             // blinded input

			const __m256 if_geq  = _mm256_cmp_ps(inp8f, mid8f, 0x0d);    // unblinded >= mid
			// const __m256 if_lt   = _mm256_cmp_ps(inp8f, negmid8f, 0x01);  // unblinded < -mid
			const __m256 then8f  = _mm256_sub_ps(inp8f, p8f);            // unblinded - p
			// const __m256 elif8f  = _mm256_add_ps(inp8f, p8f);            // unblinded + p
			const __m256 res8f = _mm256_blendv_ps(
                                        inp8f,
										then8f,
										if_geq);

			_mm256_stream_ps(&out[i], res8f);
    }
}


void ModP(MapMatRowMajor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<DtypeForCpuOp>(1) / PrimeLimit;
    m.array() = m.array() - (m * invPLimit).array() * PLimit;
}

void ModP(EigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

void ModP(MapEigenTensor& m) {
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;
    m -= (m * invPLimit).floor() * PLimit;
    // m = (m > m.constant((float) HalfPrime)).select(m - (float) HalfPrime, m);
}

class ChunkPool {
public:
    ChunkPool(int size_pool_, int num_byte_chunk_) :
        size_pool(size_pool_),
        num_byte_chunk(num_byte_chunk_)
    {
        for (int i = 0; i < size_pool; i++) {
            void* enc_chunk = (void*)memalign(64, num_byte_chunk);
            chunks.push_back(enc_chunk);
            chunk_ids.push(i);
        }
    }

    int get_chunk_id() {
        std::unique_lock<std::mutex> lock(stack_mutex);
        if (chunk_ids.empty()) {
            printf("Running out of chunks\n");
            throw std::invalid_argument("Running out of chunks");
        }
        int res;
        res = chunk_ids.top();
        chunk_ids.pop();
        return res;
    }

    void return_chunk_id(int id) {
        std::unique_lock<std::mutex> lock(stack_mutex);
        chunk_ids.push(id);
    }

    std::vector<void*> chunks;

private:
    int size_pool;
    int num_byte_chunk;
    std::mutex stack_mutex;
    std::stack<int> chunk_ids;
};

class StoreChunkPool {
public:
    static shared_ptr<ChunkPool> GetChunkPool() {
        static StoreChunkPool instance;
        return instance.chunk_pool;
    }
    StoreChunkPool(StoreChunkPool const&) = delete;
    void operator=(StoreChunkPool const&) = delete;

private:
    StoreChunkPool() {
        chunk_pool = make_shared<ChunkPool>(THREAD_POOL_SIZE * 2, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
    }
    shared_ptr<ChunkPool> chunk_pool;
};

template<typename T>
class ChunkGuard {
public:
    ChunkGuard<T>(shared_ptr<ChunkPool> chunk_pool_, T*& pointer) :
        chunk_pool(chunk_pool_)
    {
        id = chunk_pool->get_chunk_id();
        pointer = (T*) chunk_pool->chunks[id];
    }
    ~ChunkGuard<T>() {
        chunk_pool->return_chunk_id(id);
    }
private:
    int id;
    shared_ptr<ChunkPool> chunk_pool;
};


class TrustedChunkManager {
public:
    static TrustedChunkManager& getInstance() {
        static TrustedChunkManager instance;
        return instance;
    }
    TrustedChunkManager(TrustedChunkManager const&) = delete;
    void operator=(TrustedChunkManager const&) = delete;

    IdT GetNewId() {
        return id_counter++;
    }

    const int start_idx = 1000;

    void StoreChunk(IdT id, void* src_chunk, int num_byte) {
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        DtypeForCpuOp* src_float = (DtypeForCpuOp*) src_chunk;
        encrypt((uint8_t *) src_chunk,
                num_byte,
                (uint8_t *) (&(enc_chunk->payload)),
                (sgx_aes_gcm_128bit_iv_t *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)));
        DtypeForCpuOp* dst_chunk = (DtypeForCpuOp*)malloc(num_byte);
        GetChunk(id, dst_chunk, num_byte);
        uint8_t* blind_chunk;
        ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
        decrypt((uint8_t *) (&(enc_chunk->payload)),
                num_byte,
                (uint8_t *) dst_chunk,
                (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
                (uint8_t *) blind_chunk);
        src_float = (DtypeForCpuOp*) dst_chunk;
        free(dst_chunk);
    }

    void GetChunk(IdT id, void* dst_chunk, int num_byte) {
        int num_byte_enc_chunk = CalcEncDataSize(0, num_byte);
        uint8_t* blind_chunk;
        ChunkGuard<uint8_t> guard(blind_chunks, blind_chunk);
        SgxEncT* enc_chunk = (SgxEncT*) get_untrusted_mem(id, num_byte_enc_chunk);
        decrypt((uint8_t *) (&(enc_chunk->payload)),
                num_byte,
                (uint8_t *) dst_chunk,
                (sgx_aes_gcm_128bit_iv_t  *)(&(enc_chunk->reserved)),
                (sgx_aes_gcm_128bit_tag_t *)(&(enc_chunk->payload_tag)),
                (uint8_t *) blind_chunk);
        DtypeForCpuOp* src_float = (DtypeForCpuOp*) dst_chunk;
    }

private:
    TrustedChunkManager() {
        max_num_byte_plain_chunk = STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp);
        max_num_byte_enc_chunk = CalcEncDataSize(0, max_num_byte_plain_chunk);

        blind_chunks = make_shared<ChunkPool>(THREAD_POOL_SIZE, max_num_byte_plain_chunk);
    }

    void* get_untrusted_mem(IdT id, int num_byte) {
        void* dst_buf;
        bool is_diff_size = false;
        auto it = untrusted_mem_holder.begin();
        auto end = untrusted_mem_holder.end();
        int prev_num_byte;
        {
            std::unique_lock <std::mutex> lock(address_mutex);
            it = untrusted_mem_holder.find(id);
            end = untrusted_mem_holder.end();
        }
        if (it == end) {
            allocate_in_untrusted(&dst_buf, num_byte);
            {
                std::unique_lock<std::mutex> lock(address_mutex);
                untrusted_mem_holder[id] = std::make_pair(dst_buf, num_byte);
            }
        } else {
            std::unique_lock<std::mutex> lock(address_mutex);
            std::tie(dst_buf, prev_num_byte) = untrusted_mem_holder[id];
            if (prev_num_byte != num_byte) {
                is_diff_size = true;
            }
        }
        if (is_diff_size) {
			printf("id=%u\n",id);
            printf("A id has assigned with multiple size: original: %d, now: %d\n", prev_num_byte, num_byte);
            throw std::invalid_argument("A id has assigned with multiple size.");
        }
        return dst_buf;
    }

    const int size_chunk_pool = THREAD_POOL_SIZE;
    int max_num_byte_plain_chunk;
    int max_num_byte_enc_chunk;
    std::atomic<int> id_counter;
    std::mutex address_mutex;
    std::shared_ptr<ChunkPool> blind_chunks;
    std::unordered_map<int, std::pair<void*, int>> untrusted_mem_holder;
};


template <typename Func>
void run_all_chunks(Func chunk_op, int num_elem_in_chunk, int num_elem) {
    int start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk);
    }
    if (start_chunk < num_elem) chunk_op(start_chunk, num_elem - start_chunk);
}

template <typename Func>
void run_all_chunks_for_maxpool(Func chunk_op, size_t num_elem_in_chunk, size_t num_elem, size_t num_elem_out, size_t inputhw, size_t outputhw) {
    size_t start_chunk;
    for (start_chunk = 0; start_chunk + num_elem_in_chunk <= num_elem; start_chunk += num_elem_in_chunk) {
        chunk_op(start_chunk, num_elem_in_chunk, num_elem_out);
    }
    
    size_t remain_size = num_elem - start_chunk;
    if (start_chunk < num_elem) chunk_op(start_chunk, remain_size, (remain_size/inputhw)*outputhw);
}

class SecretTen {
public:
    SecretTen() {}
    SecretTen(IdT TenId_, DimsT* Dims_) : TenId(TenId_), Dims(*Dims_) { Init(); }
    ~SecretTen() { 
        for (auto& it: PrgStateHolder) free(it.second);
    }

    int GetNumElem() { return Dims.dim0 * Dims.dim1 * Dims.dim2 * Dims.dim3; }
    int GetSizeInByte() { return GetNumElem() * sizeof(DtypeForCpuOp); }

    void Init() {
        DtypeForCpuOp* store_chunk;
        ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
        auto& chunk_manager = TrustedChunkManager::getInstance();

        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = chunk_manager.GetNewId();
            ChunkIds.push_back(chunk_id);
            chunk_manager.StoreChunk(chunk_id, store_chunk, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    int GetChunkId(int start) {
        if (start >= GetNumElem()) {
            printf("The start exceed the size of the tensor.\n");
            throw std::invalid_argument("The start exceed the size of the tensor.");
        }
        return ChunkIds[start / STORE_CHUNK_ELEM];
    }

    void GetStoreChunk(int start, DtypeForCpuOp* store_chunk, int num_byte) {
        auto& chunk_manager = TrustedChunkManager::getInstance();
        int chunk_id = GetChunkId(start);
        chunk_manager.StoreChunk(chunk_id, store_chunk, num_byte * sizeof(DtypeForCpuOp));
    }

    void SetTen(DtypeForCpuOp* Arr) {
        auto& chunk_manager = TrustedChunkManager::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForCpuOp* src_arr = Arr + start;
            chunk_manager.StoreChunk(chunk_id, src_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    void GetTen(DtypeForCpuOp* Arr) {
        auto& chunk_manager = TrustedChunkManager::getInstance();
        auto chunk_op = [&](int start, int num_elem_in_op) {
            int chunk_id = GetChunkId(start);
            DtypeForCpuOp* dst_arr = Arr + start;
            chunk_manager.GetChunk(chunk_id, dst_arr, num_elem_in_op * sizeof(DtypeForCpuOp));
        };
        run_all_chunks(chunk_op, STORE_CHUNK_ELEM, GetNumElem());
    }

    void SetSeed(uint64_t RawSeed) {
        SeedT seed;
        memset(seed, 0, sizeof(SeedT));
        auto TmpRawSeed = RawSeed;
        for (int i = 0; TmpRawSeed > 0; i++) {
            seed[i] = (uint8_t) (TmpRawSeed & ((1 << 9) - 1));
            TmpRawSeed >>= 8;
        }
        PrgStateHolder[RawSeed] = (aes_stream_state*)memalign(16, sizeof(aes_stream_state));
        InitPrgWithSeed(PrgStateHolder[RawSeed], seed);
    }

    void GetRandom(DtypeForCpuOp* DstArr, uint64_t RawSeed) {
        auto PrgState = PrgStateHolder[RawSeed];
        DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
        DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float* input = DstArr + start;
            get_r(PrgState, (uint8_t*) input, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
            for(size_t j = 0; j < num_elem_in_op; j++) {
                input[j] -= floor(input[j] * invPLimit) * PLimit;
                input[j] = (input[j] >= mid) ? (input[j] - p) : input[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, GetNumElem());
    }

    void GetShare(DtypeForCpuOp* DstArr, uint64_t RawSeed) {
        auto PrgState = PrgStateHolder[RawSeed];
        const DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
        const DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

        auto& chunk_manager = TrustedChunkManager::getInstance();
//        DtypeForCpuOp* store_chunk;
//        ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
        DtypeForCpuOp* store_chunk = (DtypeForCpuOp*)memalign(64, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));


        auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
            chunk_manager.GetChunk(GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

            auto chunk_op = [&](int start, int num_elem_in_op) {
                float *input = DstArr + start_store_chunk + start;
                float *original = store_chunk + start;
                get_r(PrgState, (uint8_t *) input, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
                for (size_t j = 0; j < num_elem_in_op; j++) {
                    input[j] = original[j] - input[j];
                    input[j] -= floor(input[j] * invPLimit) * PLimit;
                    input[j] = (input[j] >= mid) ? (input[j] - p) : input[j];
                }
            };
            run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
        };
        run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, GetNumElem());

        free(store_chunk);
    }

    IdT TenId;
    DimsT Dims;
    vector<int> ChunkIds;
    unordered_map<uint64_t, aes_stream_state*> PrgStateHolder;
};

unordered_map<IdT, shared_ptr<SecretTen>> SecretTenHolder;
unordered_map<IdT, shared_ptr<EigenTensor>> TensorHolder;

shared_ptr<SecretTen> GetTenById(IdT TenId) {
    return SecretTenHolder[TenId];
}

unordered_map<uint64_t, DtypeForCpuOp> quantize_exp;

static inline float uint32_to_float(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = UINT32_C(0x7F) << 23 | x >> 9  };
    return u.d - 1.0f;
}

static inline float float_to_uniform(uint32_t x) {
    const union { uint32_t i; float d;  } u = { .i = (((UINT32_C(0x7F) << 23) | x) << 2) >> 2 };
    return u.d - 1.0f;
}

// http://prng.di.unimi.it/
class Xoshiro256 {
public:
    Xoshiro256() {}
    Xoshiro256(uint64_t raw_seed) {
        set_seed(raw_seed);
    }

    void set_seed(uint64_t raw_seed) {
        s[0] = raw_seed;
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void) {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

    void rand_like(float* arr, uint64_t n_elem) {
        if (n_elem % 2 != 0) {
            printf("n_elem has to be even.\n");
            throw string("n_elem has to be even.");
        }
        for (int i = 0; i < n_elem; i+=2) {
            const uint64_t rnd = next();
            const uint32_t b = rnd & ((((uint64_t) 1) << 32) - 1);
            const uint32_t a = rnd >> 32;
            arr[i]   = uint32_to_float(a);
            arr[i+1] = uint32_to_float(b);
        }
    }

    uint64_t s[4] = {};
};

class Xoshiro128 {
public:
    Xoshiro128() {}
    Xoshiro128(uint64_t raw_seed) {
        set_seed(raw_seed);
    }

    void set_seed(uint64_t raw_seed) {
        s[0] = raw_seed;
    }

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t next(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 + s1, 17) + s0;

        s1 ^= s0;
        s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
        s[1] = rotl(s1, 28); // c

        return result;
    }

    uint64_t s[2] = {};
};

unordered_map<uint64_t, shared_ptr<Xoshiro256>> fast_rngs;
//unordered_map<uint64_t, shared_ptr<Xoshiro128>> fast_rngs;

shared_ptr<Xoshiro256> get_fast_rng(uint64_t tag) {
    if (fast_rngs.find(tag) == fast_rngs.end()) {
        fast_rngs[tag] = make_shared<Xoshiro256>(tag);
    }
    return fast_rngs[tag];
}

void quantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten, uint64_t quantize_tag) {
    const int bits = 8;
    const int ebit = 8;
    const DtypeForCpuOp lower_limit = -pow(2, (bits - 1));
    const DtypeForCpuOp upper_limit = pow(2, (bits - 1)) - 1;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);
    //DtypeForCpuOp max_entry = 0;
    
	const __m256 neg8f = _mm256_set1_ps(-0.0f);
    __m256 tmp8f = _mm256_set1_ps(0.0f);

    auto get_max_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        int chunk_id = src_ten->GetChunkId(start_store_chunk);
        chunk_manager.GetChunk(chunk_id, store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&store_chunk[i]);
            const __m256 abs8f = _mm256_andnot_ps(neg8f, inp8f);
            const __m256 if_eq = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
            tmp8f = _mm256_blendv_ps(tmp8f, inp8f, if_eq);
        }
        //MapEigenVector src_vecmap(store_chunk, num_elem_in_store_chunk);
        //max_entry = std::max(max_entry, src_vecmap.cwiseAbs().maxCoeff());
    };
    run_all_chunks(get_max_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
    _mm256_stream_ps(dst_store_chunk, tmp8f);
    for(int i=4;i>0;i=i>>1){
        copy(dst_store_chunk+i,dst_store_chunk+2*i,dst_store_chunk+8);
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        const __m256 inp8f2 = _mm256_load_ps(&dst_store_chunk[8]);
        const __m256 if_eq = _mm256_cmp_ps(inp8f, inp8f2, 0x0e);
        const __m256 res8f = _mm256_blendv_ps(inp8f2, inp8f, if_eq);
        _mm256_stream_ps(dst_store_chunk, res8f);
    }

    if(1){
        dst_store_chunk[0] = (dst_store_chunk[0] == 0) ? 0: floor(log2(dst_store_chunk[0]));
        const __m256 inp8f = _mm256_load_ps(dst_store_chunk);
        //tmp8f = _mm256_set1_ps(pow(-2, (ebit - 1)));
        //__m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        //__m256 res8f = _mm256_blendv_ps(tmp8f, inp8f, if_gt); 
        tmp8f = _mm256_set1_ps(pow(2, (ebit - 1)) - 1);
        __m256 if_gt = _mm256_cmp_ps(inp8f, tmp8f, 0x0e);
        tmp8f = _mm256_blendv_ps(inp8f, tmp8f, if_gt);
        _mm256_stream_ps(dst_store_chunk, tmp8f);
    }
    DtypeForCpuOp exp = dst_store_chunk[0];

  //  DtypeForCpuOp exp = (max_entry == 0) ? 0 : floor(log2(max_entry));
  //  exp = std::min(std::max(exp, (DtypeForCpuOp) pow(-2, (ebit - 1))), (DtypeForCpuOp) pow(2, (ebit - 1) - 1));
    quantize_exp[quantize_tag] = exp;
    DtypeForCpuOp enlarge_factor = pow(2, -exp + (bits - 2));

    auto& xor_rnd = *get_fast_rng(quantize_tag);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float *input = store_chunk + start;
            float *output = dst_store_chunk + start;
            xor_rnd.rand_like(output, num_elem_in_op);
            for(uint64_t i=0;i<num_elem_in_op;i+=8){
				tmp8f = _mm256_set1_ps(enlarge_factor); 
                const __m256 inp8f = _mm256_load_ps(&input[i]);
                const __m256 out8f = _mm256_load_ps(&output[i]);
                const __m256 mul8f  = _mm256_mul_ps(inp8f, tmp8f);  
                const __m256 add8f = _mm256_add_ps(mul8f, out8f);  
                const __m256 flo8f = _mm256_floor_ps(add8f);
                tmp8f = _mm256_set1_ps(lower_limit);
                __m256 if_gt = _mm256_cmp_ps(flo8f, tmp8f, 0x0e);
                __m256 res8f = _mm256_blendv_ps(tmp8f, flo8f, if_gt);
                tmp8f = _mm256_set1_ps(upper_limit);
                if_gt = _mm256_cmp_ps(res8f, tmp8f, 0x0e);
                res8f = _mm256_blendv_ps(res8f, tmp8f, if_gt);
                _mm256_stream_ps(&output[i], res8f);
            }
            //MapEigenTensor in_map = MapEigenTensor(input, 1, 1, 1, num_elem_in_op);
            //MapEigenTensor out_map = MapEigenTensor(output, 1, 1, 1, num_elem_in_op);
            //out_map = (in_map * enlarge_factor + out_map).floor().cwiseMax(lower_limit).cwiseMin(upper_limit);
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
		//add
		chunk_manager.StoreChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		//add
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}

void dequantize_stochastic(shared_ptr<SecretTen> src_ten, shared_ptr<SecretTen> dst_ten,
        uint64_t x_tag, uint64_t y_tag) {
    const int bits = 8;
    DtypeForCpuOp x_exp = quantize_exp[x_tag];
    DtypeForCpuOp y_exp = quantize_exp[y_tag];

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *store_chunk, *dst_store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);
    ChunkGuard<DtypeForCpuOp> dst_guard(StoreChunkPool::GetChunkPool(), dst_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(src_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(dst_ten->GetChunkId(start_store_chunk), dst_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        MapEigenTensor src_map = MapEigenTensor(store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        MapEigenTensor dst_map = MapEigenTensor(dst_store_chunk, 1, 1, 1, num_elem_in_store_chunk);
        DtypeForCpuOp shrink_factor = pow(2, x_exp - (bits - 2) + y_exp - (bits - 2));

        dst_map = src_map * shrink_factor;
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, src_ten->GetNumElem());
}

DtypeForCpuOp* get_small_chunk(
        shared_ptr<SecretTen> tensor,
        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {

    int size_in_byte = tensor->GetSizeInByte();
    DtypeForCpuOp* arr = (DtypeForCpuOp*) memalign(64, size_in_byte);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    chunk_manager.GetChunk(tensor->GetChunkId(0), arr, size_in_byte);
    small_chunks.emplace_back(tensor, arr);
    return arr;
}

void store_small_chunks(vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>>& small_chunks) {
    for (auto& x : small_chunks) {
        auto tensor = x.first;
        auto arr = x.second;
        auto& chunk_manager = TrustedChunkManager::getInstance();
        int size_in_byte = tensor->GetSizeInByte();
        chunk_manager.StoreChunk(tensor->GetChunkId(0), arr, size_in_byte);
        free(arr);
    }
}


class BatchnormBuffer {
public:
    BatchnormBuffer(){}
    BatchnormBuffer(IdT FunId_) : FunId(FunId_) {
        NumBatchesTrackedArr = 0;
        BackwardState = false;
    }

    ~BatchnormBuffer() = default;

    void init(
            IdT input, IdT output, IdT gamma, IdT beta,
            IdT der_input, IdT der_output, IdT der_gamma, IdT der_beta,
            IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
            IdT mu,
            uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
            int affine_, int is_cumulative_, float momentum_, float epsilon_) {

        input_tensor = GetTenById(input);
        output_tensor = GetTenById(output);
        der_input_tensor = GetTenById(der_input);
        der_output_tensor = GetTenById(der_output);
        mu_tensor = GetTenById(mu);

        // size = num_channel * sizeof(byte)
        gamma_tensor = GetTenById(gamma);
        beta_tensor = GetTenById(beta);
        der_gamma_tensor = GetTenById(der_gamma);
        der_beta_tensor = GetTenById(der_beta);
        run_mean_tensor = GetTenById(run_mean);
        run_var_tensor = GetTenById(run_var);
        cur_mean_tensor = GetTenById(cur_mean);
        cur_var_tensor = GetTenById(cur_var);

        batch = batch_;
        channel = channel_;
        height = height_;
        width = width_;
        Affine = affine_;
        momentum = momentum_;
        epsilon = epsilon_;
        is_cumulative = is_cumulative_;

        num_rows = channel * height * width;
        num_rows_in_channel = height * width;
        total_n = height * width * batch;
        default_num_batches_per_chunk = std::min(STORE_CHUNK_ELEM, input_tensor->GetNumElem()) / num_rows;

        if (STORE_CHUNK_ELEM % num_rows != 0)  {
            printf("STORE_CHUNK_ELEM % num_rows != 0\n");
            return;
        }
    }

    DtypeForCpuOp get_fraction_bag(int num_elem_in_chunk) {
        int batch_in_chunk = num_elem_in_chunk / num_rows;
        return ((DtypeForCpuOp) batch_in_chunk / batch);
    }

    int get_num_batches_per_chunk(int num_elem_in_chunk) {
        return num_elem_in_chunk / num_rows;
    }

    void forward(int training) {
        Training = training;

        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>> small_chunks;

        auto& chunk_manager = TrustedChunkManager::getInstance();
        DtypeForCpuOp *data_chunk, *mu_chunk;
        ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
        ChunkGuard<DtypeForCpuOp> mu_guard(StoreChunkPool::GetChunkPool(), mu_chunk);

        EigenMatrixMap data_mat(data_chunk, num_rows, default_num_batches_per_chunk);
        EigenMatrixMap mu_mat(mu_chunk, num_rows, default_num_batches_per_chunk);

        DtypeForCpuOp *gamma_chunk = get_small_chunk(gamma_tensor, small_chunks);
        DtypeForCpuOp *beta_chunk = get_small_chunk(beta_tensor, small_chunks);
        DtypeForCpuOp *run_mean_chunk = get_small_chunk(run_mean_tensor, small_chunks);
        DtypeForCpuOp *run_var_chunk = get_small_chunk(run_var_tensor, small_chunks);
        DtypeForCpuOp *cur_mean_chunk = get_small_chunk(cur_mean_tensor, small_chunks);
        DtypeForCpuOp *cur_var_chunk = get_small_chunk(cur_var_tensor, small_chunks);

        if (training) {
            NumBatchesTrackedArr += 1;
            const DtypeForCpuOp chosen_momentum = (is_cumulative) ? (1 / (DtypeForCpuOp) NumBatchesTrackedArr) : momentum;

            fill(cur_mean_chunk, cur_mean_chunk + channel, 0);
            fill(cur_var_chunk, cur_var_chunk + channel, epsilon);

            run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
                int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
                int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

                for(uint32_t i = 0; i < channel; i++) {
                    auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                    cur_mean_chunk[i] += data_block.mean() * get_fraction_bag(num_elem_in_store_chunk);
                }
            }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

            run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
                int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
                int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

                for(uint32_t i = 0; i < channel; i++) {
                    auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                    auto mu_block = mu_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                    mu_block = data_block.array() - cur_mean_chunk[i];
                    cur_var_chunk[i] += (mu_block).cwiseProduct(mu_block).mean() * get_fraction_bag(num_elem_in_store_chunk);
                }

                chunk_manager.StoreChunk(mu_tensor->GetChunkId(start_store_chunk), mu_chunk, chunk_size_in_byte);
            }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

            run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
                int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
                int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                chunk_manager.GetChunk(mu_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

                for(uint32_t i = 0; i < channel; i++) {
                    auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                    if (Affine) {
                        data_block = (data_block.array() / sqrt(cur_var_chunk[i])) * gamma_chunk[i] + beta_chunk[i];
                    } else {
                        data_block = data_block / sqrt(cur_var_chunk[i]);
                    }
                }

                chunk_manager.StoreChunk(output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

            for (int i = 0; i < channel; i++) {
                run_mean_chunk[i] = (cur_mean_chunk[i] - run_mean_chunk[i]) * chosen_momentum + run_mean_chunk[i];
                run_var_chunk[i] = (cur_var_chunk[i] - run_var_chunk[i]) * chosen_momentum + run_var_chunk[i];
            }
        } else {
            run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
                int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
                int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
                chunk_manager.GetChunk(input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);

                for(uint32_t i = 0; i < channel; i++) {
                    auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                    data_block = data_block.array() - run_mean_chunk[i];
                    if (Affine) {
                        data_block = (data_block.array() / sqrt(run_var_chunk[i])) * gamma_chunk[i] + beta_chunk[i];
                    } else {
                        data_block = data_block / sqrt(run_var_chunk[i]);
                    }
                }

                chunk_manager.StoreChunk(output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());
        }

        store_small_chunks(small_chunks);

        BackwardState = true;
    }

    void backward() {
        if (!BackwardState) {
            printf("Forward Batch Normalization has not been done.\n");
            return;
        }

        vector<std::pair<shared_ptr<SecretTen>, DtypeForCpuOp*>> small_chunks;

        auto& chunk_manager = TrustedChunkManager::getInstance();
        DtypeForCpuOp *data_chunk, *mu_chunk;
        ChunkGuard<DtypeForCpuOp> data_guard(StoreChunkPool::GetChunkPool(), data_chunk);
        ChunkGuard<DtypeForCpuOp> mu_guard(StoreChunkPool::GetChunkPool(), mu_chunk);

        EigenMatrixMap data_mat(data_chunk, num_rows, default_num_batches_per_chunk);
        EigenMatrixMap mu_mat(mu_chunk, num_rows, default_num_batches_per_chunk);

        DtypeForCpuOp *gamma_chunk = get_small_chunk(gamma_tensor, small_chunks);
        DtypeForCpuOp *beta_chunk = get_small_chunk(beta_tensor, small_chunks);
        DtypeForCpuOp *der_gamma_chunk = get_small_chunk(der_gamma_tensor, small_chunks);
        DtypeForCpuOp *der_beta_chunk = get_small_chunk(der_beta_tensor, small_chunks);
        DtypeForCpuOp *run_mean_chunk = get_small_chunk(run_mean_tensor, small_chunks);
        DtypeForCpuOp *run_var_chunk = get_small_chunk(run_var_tensor, small_chunks);
        DtypeForCpuOp *cur_mean_chunk = get_small_chunk(cur_mean_tensor, small_chunks);
        DtypeForCpuOp *cur_var_chunk = get_small_chunk(cur_var_tensor, small_chunks);

        fill(der_beta_chunk, der_beta_chunk + channel, 0);
        fill(der_gamma_chunk, der_gamma_chunk + channel, 0);

        run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
            int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
            int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            chunk_manager.GetChunk(der_output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            chunk_manager.GetChunk(mu_tensor->GetChunkId(start_store_chunk), mu_chunk, chunk_size_in_byte);

            for(uint32_t i = 0; i < channel; i++) {
                auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                auto mu_block = mu_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                DtypeForCpuOp variance = (Training) ? cur_var_chunk[i] : run_var_chunk[i];

                der_gamma_chunk[i] += mu_block.cwiseProduct(data_block).sum() / sqrt(variance);
                der_beta_chunk[i] += data_block.sum();
            }
        }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        run_all_chunks([&](int start_store_chunk, int num_elem_in_store_chunk) {
            int num_batches_per_chunk = get_num_batches_per_chunk(num_elem_in_store_chunk);
            int chunk_size_in_byte = num_elem_in_store_chunk * sizeof(DtypeForCpuOp);
            chunk_manager.GetChunk(der_output_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
            chunk_manager.GetChunk(mu_tensor->GetChunkId(start_store_chunk), mu_chunk, chunk_size_in_byte);

            for(uint32_t i = 0; i < channel; i++) {
                auto data_block = data_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                auto mu_block = mu_mat.block(i * num_rows_in_channel, 0, num_rows_in_channel, num_batches_per_chunk);
                DtypeForCpuOp variance = (Training) ? cur_var_chunk[i] : run_var_chunk[i];
                DtypeForCpuOp gamma = (Affine) ? gamma_chunk[i] : 1;

                mu_block *= der_gamma_chunk[i] / sqrt(variance);
                variance = sqrt(variance);
                // der_gamma_chunk[i] /= variance;
                variance = gamma / ((DtypeForCpuOp) total_n * variance);
                data_block = total_n * data_block.array() - der_beta_chunk[i] - mu_block.array();
                data_block *= variance;
            }

            chunk_manager.StoreChunk(der_input_tensor->GetChunkId(start_store_chunk), data_chunk, chunk_size_in_byte);
        }, STORE_CHUNK_ELEM, input_tensor->GetNumElem());

        store_small_chunks(small_chunks);

        BackwardState = false;
    }
    
    IdT FunId;
    int batch;
    int channel;
    int height;
    int width;
    DtypeForCpuOp momentum;
    DtypeForCpuOp epsilon;

    bool is_cumulative;
    bool BackwardState;
    bool Affine;
    bool Training;

    int num_rows;
    int num_rows_in_channel;
    int total_n;
    int default_num_batches_per_chunk;

    int NumBatchesTrackedArr = 0;

    shared_ptr<SecretTen> input_tensor;
    shared_ptr<SecretTen> output_tensor;
    shared_ptr<SecretTen> der_input_tensor;
    shared_ptr<SecretTen> der_output_tensor;
    shared_ptr<SecretTen> mu_tensor;
    shared_ptr<SecretTen> gamma_tensor;
    shared_ptr<SecretTen> beta_tensor;
    shared_ptr<SecretTen> der_gamma_tensor;
    shared_ptr<SecretTen> der_beta_tensor;
    shared_ptr<SecretTen> run_mean_tensor;
    shared_ptr<SecretTen> run_var_tensor;
    shared_ptr<SecretTen> cur_mean_tensor;
    shared_ptr<SecretTen> cur_var_tensor;
};

class MaxpoolBuffer {
public:
    MaxpoolBuffer() {}
    MaxpoolBuffer(IdT FunId_, IdT TenIdin_trans_, IdT TenIdout_trans_) : FunId(FunId_), TenIdin_trans(TenIdin_trans_), TenIdout_trans(TenIdout_trans_)  { }

    ~MaxpoolBuffer() = default;

	IdT get_TenIdin_trans(){
		return TenIdin_trans;
	}

	IdT get_TenIdout_trans(){
		return TenIdout_trans;
	}
    //if NCHW->WHCN N=CN M=HW
    void transpose(const DtypeForCpuOp *src, DtypeForCpuOp *dst, const size_t N, const size_t M) {
    #pragma omp parallel for
       for(size_t n = 0; n<N*M; n++) {
          size_t i = n/N;
          size_t j = n%N;
          dst[n] = src[M*j + i]; 
       }   
    }

    inline void transpose4x4_SSE(const float *A, float *B, const uint32_t lda, const uint32_t ldb) {
        __m128 row1 = _mm_load_ps(&A[0*lda]);
        __m128 row2 = _mm_load_ps(&A[1*lda]);
        __m128 row3 = _mm_load_ps(&A[2*lda]);
        __m128 row4 = _mm_load_ps(&A[3*lda]);
         _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
         _mm_store_ps(&B[0*ldb], row1);
         _mm_store_ps(&B[1*ldb], row2);
         _mm_store_ps(&B[2*ldb], row3);
         _mm_store_ps(&B[3*ldb], row4);
    }

    inline void transpose_block_SSE4x4(const float *A, float *B, const uint32_t lda, const uint32_t ldb ,const int block_size) {
        #pragma omp parallel for
        for(uint32_t i=0; i<ldb; i+=block_size) {
            for(uint32_t j=0; j<lda; j+=block_size) {
                uint32_t max_i2 = i+block_size < ldb ? i + block_size : ldb;
                uint32_t max_j2 = j+block_size < lda ? j + block_size : lda;
                for(uint32_t i2=i; i2<max_i2; i2+=4) {
                    for(uint32_t j2=j; j2<max_j2; j2+=4) {
                        transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                    }
                }
            }
         }
    }
    
    inline void MaxpoolAVX(const uint32_t num_img, float* input, float* output){
        #pragma omp parallel for        
        for(size_t i=0; i<num_img; i+=8){
            const __m256 inp8f = _mm256_load_ps(&input[i]);
            const __m256 out8f = _mm256_load_ps(&output[i]);
            const __m256 if_lq = _mm256_cmp_ps(out8f, inp8f, 0x01);
            const __m256 res8f = _mm256_blendv_ps(out8f, inp8f, if_lq);
            _mm256_stream_ps(&output[i], res8f);
        }
    }

    inline void MaxpoolbackAVX(const uint32_t num_img, float* input, float* output, float* dinput, float* doutput){
        #pragma omp parallel for
        for(size_t i=0; i<num_img; i+=8){
            const __m256 inp8f = _mm256_load_ps(&input[i]);
            const __m256 out8f = _mm256_load_ps(&output[i]);
            const __m256 din8f = _mm256_load_ps(&dinput[i]);
            const __m256 dout8f = _mm256_load_ps(&doutput[i]);
            const __m256 if_eq = _mm256_cmp_ps(out8f, inp8f, 0x00);
            const __m256 sum8f = _mm256_add_ps(din8f, dout8f);
            const __m256 res8f = _mm256_blendv_ps(din8f, sum8f, if_eq); // define dinput
            const __m256 res28f = _mm256_blendv_ps(dout8f, zero8f, if_eq); // redefine doutput
            _mm256_store_ps(&dinput[i], res8f);
            _mm256_stream_ps(&doutput[i], res28f);
        }
    }

    void forward(
           shared_ptr<SecretTen> ten_in, shared_ptr<SecretTen> ten_out,
           shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
           uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
           uint32_t output_height, uint32_t output_width, uint32_t filter_height,
           uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {

        const uint32_t inputhw = input_height*input_width;
        const uint32_t num_img_in_storechunk = STORE_CHUNK_ELEM/inputhw;

		if(STORE_CHUNK_ELEM % inputhw != 0){
			printf("STORE_CHUNK_ELEM %% inputhw != 0\n");
			return;
		}
		//if (num_img_in_storechunk % 8 != 0){
		//	printf("STORE_CHUNK_ELEM/inputhw is not divisible by 8!\n");
		//	return;
		//}

        const uint32_t outputhw = output_height * output_width;
        uint32_t outputsize_in_storechunk = num_img_in_storechunk * outputhw;
        const uint32_t total_size = batch * channel * inputhw;
        size_t idx_out=0;
        size_t idx_tmp=0;
		size_t size_of_store_chunk = STORE_CHUNK_ELEM * sizeof(float);      
        bool if_use_SSE_out =(outputhw%4==0);

        float* chunk_in, *chunk_out, *chunk_in_trans, *chunk_out_trans, *chunk_tmp;
		auto& chunk_manager = TrustedChunkManager::getInstance();

        ChunkGuard<DtypeForCpuOp> guard_in(StoreChunkPool::GetChunkPool(), chunk_in);
        ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
        ChunkGuard<DtypeForCpuOp> guard_int(StoreChunkPool::GetChunkPool(), chunk_in_trans);
        ChunkGuard<DtypeForCpuOp> guard_outt(StoreChunkPool::GetChunkPool(), chunk_out_trans);
        ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp); // chunk_tmp is used to store output temporarily
      
        auto chunk_op = [&](size_t start_chunk, size_t num_elem_in, size_t num_elem_out) {
            // printf("maxpooling forward in enclave. start_chunk: %d\n", start_chunk);
            chunk_manager.GetChunk(ten_in->GetChunkId(start_chunk), chunk_in, num_elem_in * sizeof(DtypeForCpuOp));
            transpose_block_SSE4x4(chunk_in, chunk_in_trans, inputhw, num_img_in_storechunk, 8);
            chunk_manager.StoreChunk(ten_in_trans->GetChunkId(start_chunk), chunk_in_trans, size_of_store_chunk);
			fill(chunk_out_trans, chunk_out_trans + outputsize_in_storechunk, std::numeric_limits<DtypeForCpuOp>::lowest());
            for(uint32_t h = 0; h < input_height; ++h) {
                for(uint32_t w = 0; w < input_width; ++w) {
                    // (h_start, h_end) * (w_start, w_end) is the range that the input
                    // vector projects to.
                    const uint32_t h_start = (h < filter_height)
                                            ? 0
                                            : (h - filter_height) / row_stride + 1;
                    const uint32_t h_end = std::min(h / row_stride + 1, output_height);
                    const uint32_t w_start = (w < filter_width)
                                            ? 0
                                            : (w - filter_width) / col_stride + 1;
                    const uint32_t w_end = std::min(w / col_stride + 1, output_width);
                    // compute elementwise max
                    const uint32_t in_offset = (h * input_width + w)*num_img_in_storechunk;
                    for (uint32_t ph = h_start; ph < h_end; ++ph) {
                        const uint32_t out_offset_base = ph * output_width;
                        for (uint32_t pw = w_start; pw < w_end; ++pw) {
                            const uint32_t out_offset = (out_offset_base + pw) * num_img_in_storechunk;
                            MaxpoolAVX(num_img_in_storechunk, chunk_in_trans+in_offset, chunk_out_trans + out_offset);
                        }
                    }
                }
            }
            chunk_manager.StoreChunk(ten_out_trans->GetChunkId(start_chunk), chunk_out_trans, size_of_store_chunk);
            //transpose
            if(if_use_SSE_out){
                transpose_block_SSE4x4(chunk_out_trans, chunk_tmp, num_img_in_storechunk, outputhw, 8);
            }
            else{
                transpose(chunk_out_trans, chunk_tmp, outputhw, num_img_in_storechunk);
            }
            if(idx_tmp+num_elem_out<STORE_CHUNK_ELEM){
                copy(chunk_tmp, chunk_tmp+num_elem_out, chunk_out + idx_tmp);
                idx_tmp+=num_elem_out;
            }
            else{
                size_t idx_add = STORE_CHUNK_ELEM-idx_tmp;
                copy(chunk_tmp,chunk_tmp+idx_add,chunk_out+idx_tmp);
                chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, size_of_store_chunk);
                idx_out += STORE_CHUNK_ELEM;
                copy(chunk_tmp + idx_add,chunk_tmp + num_elem_out,chunk_out + idx_tmp+idx_add);
                idx_tmp += num_elem_out;
				idx_tmp -= STORE_CHUNK_ELEM; 
            }
        };//end of chunk_op
        run_all_chunks_for_maxpool(chunk_op, STORE_CHUNK_ELEM, batch * channel * inputhw, outputsize_in_storechunk, inputhw, outputhw);      

        if (idx_tmp!=0) {
            chunk_manager.StoreChunk(ten_out->GetChunkId(idx_out), chunk_out, idx_tmp * sizeof(DtypeForCpuOp)); 
        }
	}//end maxpooling

    void backward(
            shared_ptr<SecretTen> ten_din, shared_ptr<SecretTen> ten_dout,
            shared_ptr<SecretTen> ten_in_trans, shared_ptr<SecretTen> ten_out_trans,
            uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,
            uint32_t output_height, uint32_t output_width,
            uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride) {

        const uint32_t num_img = batch*channel;
        const uint32_t inputhw = input_height * input_width;
        const uint32_t num_img_in_storechunk = STORE_CHUNK_ELEM / inputhw;
        const uint32_t outputhw = output_height*output_width;
        uint32_t outputsize_in_storechunk = num_img_in_storechunk * outputhw;
        const uint32_t total_size = num_img * inputhw;
        const uint32_t total_size_out = num_img * outputhw;

        size_t idx_dout=0;
        size_t idx_tmp=0;
        bool if_use_SSE_out = (outputhw%4==0);
        float* chunk_din, *chunk_dout, *chunk_in_trans, *chunk_out_trans, *chunk_din_trans, *chunk_dout_trans, *chunk_tmp;
		auto& chunk_manager = TrustedChunkManager::getInstance();

        ChunkGuard<DtypeForCpuOp> guard_din(StoreChunkPool::GetChunkPool(), chunk_din);
        ChunkGuard<DtypeForCpuOp> guard_dout(StoreChunkPool::GetChunkPool(), chunk_dout);
        ChunkGuard<DtypeForCpuOp> guard_int(StoreChunkPool::GetChunkPool(), chunk_in_trans);
        ChunkGuard<DtypeForCpuOp> guard_outt(StoreChunkPool::GetChunkPool(), chunk_out_trans);
		ChunkGuard<DtypeForCpuOp> guard_dint(StoreChunkPool::GetChunkPool(), chunk_din_trans);
		ChunkGuard<DtypeForCpuOp> guard_doutt(StoreChunkPool::GetChunkPool(), chunk_dout_trans);
        ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);

        size_t start_chunk_out=0;
        if(total_size>=STORE_CHUNK_ELEM){
			size_t getsize_out;
			if(STORE_CHUNK_ELEM>total_size_out){
				getsize_out = total_size_out;
			}
			else{
				getsize_out = STORE_CHUNK_ELEM;
			}
            chunk_manager.GetChunk(ten_dout->GetChunkId(0), chunk_tmp, getsize_out * sizeof(DtypeForCpuOp));
            start_chunk_out += getsize_out; 
        }
        else{
            chunk_manager.GetChunk(ten_dout->GetChunkId(0), chunk_tmp, total_size_out * sizeof(float));
        }
        auto chunk_op = [&](size_t start_chunk, size_t num_elem_in, size_t num_elem_out) {
            chunk_manager.GetChunk(ten_in_trans->GetChunkId(start_chunk), chunk_in_trans, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
            chunk_manager.GetChunk(ten_out_trans->GetChunkId(start_chunk), chunk_out_trans, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
            if(num_elem_in == STORE_CHUNK_ELEM){    
                if(idx_tmp + outputsize_in_storechunk > STORE_CHUNK_ELEM){
                    copy(chunk_tmp+idx_tmp,chunk_tmp+STORE_CHUNK_ELEM,chunk_dout);
                    idx_dout = STORE_CHUNK_ELEM-idx_tmp;
                    chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_tmp, STORE_CHUNK_ELEM * sizeof(DtypeForCpuOp));
        			start_chunk_out += STORE_CHUNK_ELEM;
                    idx_tmp = outputsize_in_storechunk-idx_dout;
                    copy(chunk_tmp, chunk_tmp+idx_tmp, chunk_dout+idx_dout);
                }
                else{
                    copy(chunk_tmp+idx_tmp,chunk_tmp+idx_tmp+outputsize_in_storechunk,chunk_dout);
                    idx_tmp += outputsize_in_storechunk;
                }
            }
            else{
                if(idx_tmp==STORE_CHUNK_ELEM||idx_tmp==0){
                    chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_dout, (total_size_out-start_chunk_out) * sizeof(DtypeForCpuOp));
                }
                else{
                    copy(chunk_tmp+idx_tmp,chunk_tmp+STORE_CHUNK_ELEM,chunk_dout);
                    idx_dout = STORE_CHUNK_ELEM-idx_tmp;
        			if(total_size_out!=start_chunk_out)
                        chunk_manager.GetChunk(ten_dout->GetChunkId(start_chunk_out), chunk_tmp, (total_size_out-start_chunk_out) * sizeof(DtypeForCpuOp));
                        //assume total_size_out-start_chunk_out+idx_dout<=STORE_CHUNK_ELEM
                    idx_tmp = total_size_out - start_chunk_out;
                    copy(chunk_tmp, chunk_tmp+idx_tmp, chunk_dout+idx_dout);
                        //idx_dout
                }
                
            }

            if(if_use_SSE_out){
                transpose_block_SSE4x4(chunk_dout, chunk_dout_trans, outputhw, num_img_in_storechunk, 4);
            }
            else{
                transpose(chunk_dout, chunk_dout_trans, num_img_in_storechunk, outputhw);
            }
    		fill(chunk_din_trans, chunk_din_trans + STORE_CHUNK_ELEM,0);
            for(uint32_t h = 0; h < input_height; ++h) {
                for(uint32_t w = 0; w < input_width; ++w) {
                    // (h_start, h_end) * (w_start, w_end) is the range that the input
                    // vector projects to.
                    const uint32_t h_start = (h < filter_height)
                                            ? 0
                                            : (h - filter_height) / row_stride + 1;
                    const uint32_t h_end = std::min(h / row_stride + 1, output_height);
                    const uint32_t w_start = (w < filter_width)
                                            ? 0
                                            : (w - filter_width) / col_stride + 1;
                    const uint32_t w_end = std::min(w / col_stride + 1, output_width);
                    // compute elementwise max
                    const uint32_t in_offset = (h * input_width + w)*num_img_in_storechunk;
                    for (uint32_t ph = h_start; ph < h_end; ++ph) {
                        const uint32_t out_offset_base = ph * output_width;
                        for (uint32_t pw = w_start; pw < w_end; ++pw) {
                            const uint32_t out_offset = (out_offset_base + pw) * num_img_in_storechunk;
                            MaxpoolbackAVX(num_img_in_storechunk, chunk_in_trans + in_offset, chunk_out_trans + out_offset, chunk_din_trans + in_offset, chunk_dout_trans + out_offset);
                        }
                    }
                }
            }
            //transpose
            transpose_block_SSE4x4(chunk_din_trans, chunk_din, num_img_in_storechunk ,inputhw, 8);
            chunk_manager.StoreChunk(ten_din->GetChunkId(start_chunk), chunk_din, num_elem_in * sizeof(float));
        };//end of chunk_op
        run_all_chunks_for_maxpool(chunk_op, STORE_CHUNK_ELEM, total_size, outputsize_in_storechunk, inputhw, outputhw);
	}//end maxpoolbackward

    IdT FunId;
	IdT TenIdin_trans;
   	IdT TenIdout_trans;
};

static inline float float2_to_uniform(uint32_t x, uint32_t y, float& a, float& b) {
    const union { uint32_t i; float d;  } u = { .i = UINT32_C(0x7F) << 23 | ((x ^ y) >> 2) };
    const union { uint32_t i; float d;  } v = { .i = UINT32_C(0x7F) << 23 | (((x ^ y) >> 5) ^ UINT32_C(0x7FFFFF))};
    a = u.d - 1.0f;
    b = v.d - 1.0f;
}

// Input: Af
// Output: E
// E = AQ - U = Q(Af) - U
// test: E + U ~= Q(Af)
//void FusedQuantizeShare(shared_ptr<SecretTen> af_ten, shared_ptr<SecretTen> e_ten, uint64_t q_tag, uint64_t u_seed) {
void FusedQuantizeShare(shared_ptr<SecretTen> af_ten, DtypeForCpuOp* e_arr, uint64_t q_tag, uint64_t u_seed) {
    const int bits = 8;
    const int ebit = 8;
    const DtypeForCpuOp lower_limit = -pow(2, (bits - 1));
    const DtypeForCpuOp upper_limit = pow(2, (bits - 1)) - 1;
    const int num_elem_in_chunk = WORK_CHUNK_ELEM;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    DtypeForCpuOp max_entry = 0;
    auto get_max_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        int chunk_id = af_ten->GetChunkId(start_store_chunk);
        chunk_manager.GetChunk(chunk_id, store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        MapEigenVector src_vecmap(store_chunk, num_elem_in_store_chunk);
        max_entry = std::max(max_entry, src_vecmap.cwiseAbs().maxCoeff());
    };
    run_all_chunks(get_max_chunk_op, STORE_CHUNK_ELEM, af_ten->GetNumElem());

    DtypeForCpuOp exp = (max_entry == 0) ? 0 : floor(log2(max_entry));
    exp = std::min(std::max(exp, (DtypeForCpuOp) pow(-2, (ebit - 1))), (DtypeForCpuOp) pow(2, (ebit - 1) - 1));
    quantize_exp[q_tag] = exp;
    const DtypeForCpuOp enlarge_factor = pow(2, -exp + (bits - 2));

    const DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    const DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    auto& xor_rnd = *get_fast_rng(q_tag);
    auto PrgState = af_ten->PrgStateHolder[u_seed];
    DtypeForCpuOp* tmp_chunk = (DtypeForCpuOp*)malloc(num_elem_in_chunk * sizeof(DtypeForCpuOp));

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(af_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float* af_chunk = store_chunk + start;
            float* e_chunk = e_arr + start_store_chunk + start;
            MapEigenTensor af_map = MapEigenTensor(af_chunk, 1, 1, 1, num_elem_in_op);
            MapEigenTensor tmp_map = MapEigenTensor(tmp_chunk, 1, 1, 1, num_elem_in_op);

            get_r(PrgState, (uint8_t*) e_chunk, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
#if QUANTIZE_MODE == STOCHASTIC
//            xor_rnd.rand_like(tmp_chunk, num_elem_in_op);
            uint32_t* uint32_chunk = (uint32_t*) e_chunk;
//            uint32_t* uint32_chunk = reinterpret_cast<uint32_t*>(e_chunk);
//            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = uint32_to_float(uint32_chunk[j]);
            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = float_to_uniform(uint32_chunk[j]);
//            for(size_t j = 0; j < 10; j++) {
//                printf("%f ", tmp_chunk[j]);
//            }
//            printf("\n");
//            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = e_chunk[j];
            tmp_map = (af_map * enlarge_factor + tmp_map).floor().cwiseMax(lower_limit).cwiseMin(upper_limit);
#else
            tmp_map = (af_map * enlarge_factor).round().cwiseMax(lower_limit).cwiseMin(upper_limit);
#endif
            for(size_t j = 0; j < num_elem_in_op; j++) {
                e_chunk[j] = tmp_chunk[j] - e_chunk[j];
                e_chunk[j] -= floor(e_chunk[j] * invPLimit) * PLimit;
                e_chunk[j] = (e_chunk[j] >= mid) ? (e_chunk[j] - p) : e_chunk[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, af_ten->GetNumElem());

    free(tmp_chunk);
}

// Input: Af
// Output: A1, E
// AQ = Q(Af)
// A0, U <- Random
// A1 = AQ - A0
// E = AQ - U
// test: E + U = A0 + A1 ~= AQ ~= Q(Af)
void FusedQuantizeShare2(shared_ptr<SecretTen> af_ten, DtypeForCpuOp* a1_arr, DtypeForCpuOp* e_arr,
        uint64_t q_tag, uint64_t a0_seed, uint64_t u_seed) {

    const int bits = 8;
    const int ebit = 8;
    const DtypeForCpuOp lower_limit = -pow(2, (bits - 1));
    const DtypeForCpuOp upper_limit = pow(2, (bits - 1)) - 1;

    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    DtypeForCpuOp max_entry = 0;
    auto get_max_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(af_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        MapEigenVector src_vecmap(store_chunk, num_elem_in_store_chunk);
        max_entry = std::max(max_entry, src_vecmap.cwiseAbs().maxCoeff());
    };
    run_all_chunks(get_max_chunk_op, STORE_CHUNK_ELEM, af_ten->GetNumElem());

    DtypeForCpuOp exp = (max_entry == 0) ? 0 : floor(log2(max_entry));
    exp = std::min(std::max(exp, (DtypeForCpuOp) pow(-2, (ebit - 1))), (DtypeForCpuOp) pow(2, (ebit - 1) - 1));
    quantize_exp[q_tag] = exp;
    DtypeForCpuOp enlarge_factor = pow(2, -exp + (bits - 2));

    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    auto& xor_rnd = *get_fast_rng(q_tag);

    const int n_elem_in_chunk = WORK_CHUNK_ELEM;

    auto u_prg_state = af_ten->PrgStateHolder[u_seed];
    auto a0_prg_state = af_ten->PrgStateHolder[a0_seed];
    DtypeForCpuOp* tmp_chunk = (DtypeForCpuOp*)malloc(n_elem_in_chunk * sizeof(DtypeForCpuOp));

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(af_ten->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            float* af_chunk = store_chunk + start;
            float* a1_chunk = a1_arr + start_store_chunk + start;
            float* e_chunk = e_arr + start_store_chunk + start;
            MapEigenTensor af_map = MapEigenTensor(af_chunk, 1, 1, 1, num_elem_in_op);
            MapEigenTensor tmp_map = MapEigenTensor(tmp_chunk, 1, 1, 1, num_elem_in_op);

            get_r(a0_prg_state, (uint8_t*) a1_chunk, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
            get_r(u_prg_state, (unsigned char*) e_chunk, num_elem_in_op * sizeof(DtypeForCpuOp), 0);
#if QUANTIZE_MODE == STOCHASTIC
//            xor_rnd.rand_like(tmp_chunk, num_elem_in_op);
            uint32_t* uint32_chunk = (uint32_t*) e_chunk;
//            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = uint32_to_float(uint32_chunk[j]);
            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = float_to_uniform(uint32_chunk[j]);
//            for(size_t j = 0; j < num_elem_in_op; j++) tmp_chunk[j] = e_chunk[j];
            tmp_map = (af_map * enlarge_factor + tmp_map).floor().cwiseMax(lower_limit).cwiseMin(upper_limit);
#else
            tmp_map = (af_map * enlarge_factor).round().cwiseMax(lower_limit).cwiseMin(upper_limit);
#endif
            for(size_t j = 0; j < num_elem_in_op; j++) {
                e_chunk[j] = tmp_chunk[j] - e_chunk[j];
                e_chunk[j] -= floor(e_chunk[j] * invPLimit) * PLimit;
                e_chunk[j] = (e_chunk[j] >= mid) ? (e_chunk[j] - p) : e_chunk[j];
                a1_chunk[j] = tmp_chunk[j] - a1_chunk[j];
                a1_chunk[j] -= floor(a1_chunk[j] * invPLimit) * PLimit;
                a1_chunk[j] = (a1_chunk[j] >= mid) ? (a1_chunk[j] - p) : a1_chunk[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, af_ten->GetNumElem());

    free(tmp_chunk);
    }

// Input: C', Ci
// Output: Cf
// Cf = dQ(C' + Ci)
// test: Cf ~= deQ(C' + Ci)
void FusedRecon(shared_ptr<SecretTen> cf_ten, shared_ptr<SecretTen> cq_ten, DtypeForCpuOp* c_left_arr,
        uint64_t x_tag, uint64_t y_tag) {
    const int bits = 8;

    const DtypeForCpuOp x_exp = quantize_exp[x_tag];
    const DtypeForCpuOp y_exp = quantize_exp[y_tag];
    const DtypeForCpuOp shrink_factor = pow(2, x_exp - (bits - 2) + y_exp - (bits - 2));
    const DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    const DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    const int total_num_elem = cf_ten->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *cf_store_chunk, *cq_store_chunk;
    ChunkGuard<DtypeForCpuOp> cf_guard(StoreChunkPool::GetChunkPool(), cf_store_chunk);
    ChunkGuard<DtypeForCpuOp> cq_guard(StoreChunkPool::GetChunkPool(), cq_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(cq_ten->GetChunkId(start_store_chunk), cq_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            DtypeForCpuOp* cf_chunk = cf_store_chunk + start;
            DtypeForCpuOp* cq_chunk = cq_store_chunk + start;
            DtypeForCpuOp* c_left_chunk = c_left_arr + start_store_chunk + start;

            for(size_t j = 0; j < num_elem_in_op; j++) {
                cq_chunk[j] += c_left_chunk[j];
                cf_chunk[j] = cq_chunk[j];
                cf_chunk[j] -= floor(cf_chunk[j] * invPLimit) * PLimit;
                cf_chunk[j] = (cf_chunk[j] >= mid) ? (cf_chunk[j] - p) : cf_chunk[j];
                cf_chunk[j] *= shrink_factor;
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
        // chunk_manager.StoreChunk(cq_ten->GetChunkId(start_store_chunk), cq_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.StoreChunk(cf_ten->GetChunkId(start_store_chunk), cf_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

extern "C" {

void SecretInitTensor(IdT TenId, void *voidDims) {
    DimsT *Dims = (DimsT *) voidDims;
    SecretTenHolder[TenId] = make_shared<SecretTen>(TenId, Dims);
}

void SecretSetTen(IdT TenId, void *voidArr) {
    GetTenById(TenId)->SetTen((DtypeForCpuOp *) voidArr);
}

void SecretGetTen(IdT TenId, void *voidArr) {
    GetTenById(TenId)->GetTen((DtypeForCpuOp *) voidArr);
}

void SecretSetSeed(IdT TenId, uint64_t RawSeed) {
    GetTenById(TenId)->SetSeed(RawSeed);
}

void SecretGetRandom(IdT TenId, void *voidArr, uint64_t RawSeed) {
    GetTenById(TenId)->GetRandom((DtypeForCpuOp *) voidArr, RawSeed);
}

void SecretGetShare(IdT TenId, void *voidArr, uint64_t RawSeed) {
    GetTenById(TenId)->GetShare((DtypeForCpuOp *) voidArr, RawSeed);
}

void SecretAddFromCpu(void* inputArr, IdT dstId) {
    shared_ptr<SecretTen > StoreTensor = GetTenById(dstId);
    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    const int total_num_elem = StoreTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start_chunk, int num_elem_in_op) {
            DtypeForCpuOp* output_arr = store_chunk + start_chunk;
            DtypeForCpuOp* input_arr = ((DtypeForCpuOp*) inputArr) + start_store_chunk + start_chunk;
            for(size_t j = 0; j < num_elem_in_op; j++) {
                output_arr[j] += input_arr[j];
                output_arr[j] -= floor(output_arr[j] * invPLimit) * PLimit;
                output_arr[j] = (output_arr[j] >= mid) ? (output_arr[j] - p) : output_arr[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);

        chunk_manager.StoreChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

void newrelu(IdT TenIdin, IdT TenIdout, uint64_t size){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);
	shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_in,* chunk_tmp;
    ChunkGuard<DtypeForCpuOp> guard_tmp(StoreChunkPool::GetChunkPool(), chunk_tmp);
    //ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_in->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
		for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&chunk_tmp[i]);         
            const __m256 if_gt = _mm256_cmp_ps(inp8f, zero8f, 0x0e);
            const __m256 res8f = _mm256_blendv_ps(zero8f, inp8f, if_gt);
            _mm256_stream_ps(&chunk_tmp[i], res8f);
        } 
		chunk_manager.StoreChunk(ten_out->GetChunkId(start_store_chunk), chunk_tmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);
}

void newreluback(IdT TenIdout, IdT TenIddout,IdT TenIddin, uint64_t size){
    shared_ptr<SecretTen > ten_din = GetTenById(TenIddin);
    shared_ptr<SecretTen > ten_dout = GetTenById(TenIddout);
    shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* chunk_dtmp,* chunk_out;
    //ChunkGuard<DtypeForCpuOp> guard_din(StoreChunkPool::GetChunkPool(), chunk_din);
    ChunkGuard<DtypeForCpuOp> guard_dtmp(StoreChunkPool::GetChunkPool(), chunk_dtmp);
    ChunkGuard<DtypeForCpuOp> guard_out(StoreChunkPool::GetChunkPool(), chunk_out);
    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ten_dout->GetChunkId(start_store_chunk),chunk_dtmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp)); 
        chunk_manager.GetChunk(ten_out->GetChunkId(start_store_chunk),chunk_out, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        for(uint64_t i=0;i<num_elem_in_store_chunk;i+=8){
            const __m256 inp8f = _mm256_load_ps(&chunk_out[i]);
            const __m256 if_eq = _mm256_cmp_ps(inp8f, zero8f, 0x00);
            const __m256 gra8f = _mm256_load_ps(&chunk_dtmp[i]);
            const __m256 res8f = _mm256_blendv_ps(gra8f, zero8f, if_eq);
            _mm256_stream_ps(&chunk_dtmp[i], res8f);
        }
        chunk_manager.StoreChunk(ten_din->GetChunkId(start_store_chunk), chunk_dtmp, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, size);	
}

unordered_map<IdT, shared_ptr<MaxpoolBuffer>> MaxpoolHolder;


shared_ptr<MaxpoolBuffer> GetBufferByIdM(IdT FunId) {
    return MaxpoolHolder[FunId];
}

void initmaxpool(IdT FunId, IdT TenIdin_trans, IdT TenIdout_trans){	
    MaxpoolHolder[FunId] = make_shared<MaxpoolBuffer>(FunId, TenIdin_trans, TenIdout_trans);
}

void newmaxpool(IdT FunId, IdT TenIdin, IdT TenIdout, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride, uint32_t row_pad, uint32_t col_pad){
    shared_ptr<SecretTen > ten_in = GetTenById(TenIdin);                      
    shared_ptr<SecretTen > ten_out = GetTenById(TenIdout);
	IdT TenIdin_trans = GetBufferByIdM(FunId)->get_TenIdin_trans();
	shared_ptr<SecretTen> ten_in_trans = GetTenById(TenIdin_trans);
	IdT TenIdout_trans = GetBufferByIdM(FunId)->get_TenIdout_trans();
    shared_ptr<SecretTen> ten_out_trans = GetTenById(TenIdout_trans);  
	GetBufferByIdM(FunId)->forward(ten_in, ten_out,ten_in_trans, ten_out_trans, batch, channel,input_height,input_width,output_height,output_width,filter_height,filter_width,row_stride,col_stride);
}

void newmaxpoolback(IdT FunId, IdT TenIddout,IdT TenIddin, uint32_t batch, uint32_t channel,uint32_t input_height, uint32_t input_width,uint32_t output_height, uint32_t output_width, uint32_t filter_height, uint32_t filter_width, uint32_t row_stride, uint32_t col_stride){
    shared_ptr<SecretTen > ten_din = GetTenById(TenIddin);                                                                                                
    shared_ptr<SecretTen > ten_dout = GetTenById(TenIddout);
    IdT TenIdin_trans = GetBufferByIdM(FunId)->get_TenIdin_trans();
    shared_ptr<SecretTen> ten_in_trans = GetTenById(TenIdin_trans);
    IdT TenIdout_trans = GetBufferByIdM(FunId)->get_TenIdout_trans();
    shared_ptr<SecretTen> ten_out_trans = GetTenById(TenIdout_trans);                                                                                               
    //shared_ptr<SecretTen > ten_in_trans = GetTenById(0);
    //uint64_t tensor_size=(batch*channel*input_height*input_width+STORE_CHUNK_ELEM/2)/STORE_CHUNK_ELEM*STORE_CHUNK_ELEM;
    //shared_ptr<SecretTen > ten_out_trans = GetTenById(tensor_size*sizeof(float));
    GetBufferByIdM(FunId)->backward(ten_din, ten_dout, ten_in_trans, ten_out_trans, batch, channel,input_height,input_width,output_height,output_width,filter_height,filter_width,row_stride,col_stride);
}

unordered_map<IdT, shared_ptr<BatchnormBuffer>> BatchnormHolder;
    
shared_ptr<BatchnormBuffer> GetBufferByIdB(IdT FunId) {
    return BatchnormHolder[FunId];
}
    
void SecretInitBatchnorm(
        IdT FunId,
        IdT input, IdT output, IdT gamma, IdT beta,
        IdT der_input, IdT der_output, IdT der_gamma, IdT der_beta,
        IdT run_mean, IdT run_var, IdT cur_mean, IdT cur_var,
        IdT mu,
        uint32_t batch_, uint32_t channel_, uint32_t height_, uint32_t width_,
        int affine_, int is_cumulative_, float momentum_, float epsilon_) {

    auto bn_buffer = make_shared<BatchnormBuffer>(FunId);
    BatchnormHolder[FunId] = bn_buffer;

    bn_buffer->init(
            input, output, gamma, beta,
            der_input, der_output, der_gamma, der_beta,
            run_mean, run_var, cur_mean, cur_var,
            mu,
            batch_, channel_, height_, width_,
            affine_, is_cumulative_, momentum_, epsilon_);
}

void SecretBatchnormForward(IdT FunId, int Training) {
    GetBufferByIdB(FunId)->forward(Training);
}

void SecretBatchnormBackward(IdT FunId) {
    GetBufferByIdB(FunId)->backward();
}

// Store <- C0 + C1 + C2 (MainSeed + Seed1 + Seed2)
// DstArr <- MainSeed (either C0 or C1)
void SecretMaskingC01(IdT storeId, uint64_t mainRawSeed, uint64_t rawSeed0, uint64_t rawSeed1, DtypeForCpuOp *DstArr) {
    shared_ptr<SecretTen > StoreTensor = GetTenById(storeId);

    auto MainPrgState = StoreTensor->PrgStateHolder[mainRawSeed];
    auto PrgState0 = StoreTensor->PrgStateHolder[rawSeed0];
    auto PrgState1 = StoreTensor->PrgStateHolder[rawSeed1];

    const int total_num_elem = StoreTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp* store_chunk;
    ChunkGuard<DtypeForCpuOp> guard(StoreChunkPool::GetChunkPool(), store_chunk);

    DtypeForCpuOp* aux_chunk_arr = (DtypeForCpuOp*)memalign(32, WORK_CHUNK_ELEM * sizeof(DtypeForCpuOp));

    DtypeForCpuOp PLimit = static_cast<DtypeForCpuOp>(PrimeLimit);
    DtypeForCpuOp invPLimit = static_cast<double>(1) / PrimeLimit;

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            DtypeForCpuOp* store_arr = store_chunk + start;
            DtypeForCpuOp* output_arr = DstArr + start_store_chunk + start;
            get_r(MainPrgState, (uint8_t*) output_arr, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
            get_r(PrgState0, (uint8_t*) store_arr, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
            get_r(PrgState1, (uint8_t*) aux_chunk_arr, num_elem_in_op * sizeof(DtypeForCpuOp), 9);
            for(size_t j = 0; j < num_elem_in_op; j++) {
                store_arr[j] += output_arr[j] + aux_chunk_arr[j];
                store_arr[j] -= floor(store_arr[j] * invPLimit) * PLimit;
                store_arr[j] = (store_arr[j] >= mid) ? (store_arr[j] - p) : store_arr[j];
                output_arr[j] -= floor(output_arr[j] * invPLimit) * PLimit;
                output_arr[j] = (output_arr[j] >= mid) ? (output_arr[j] - p) : output_arr[j];
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
        chunk_manager.StoreChunk(StoreTensor->GetChunkId(start_store_chunk), store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);

    free(aux_chunk_arr);
}

// Assume momentum > 0
void SecretSgdUpdate(IdT paramId, IdT gradId, IdT momentumId,
        DtypeForCpuOp lr, DtypeForCpuOp momentum, DtypeForCpuOp weight_decay,
        DtypeForCpuOp dampening, bool nesterov, bool first_momentum) {

    shared_ptr<SecretTen> ParamTensor = GetTenById(paramId);
    shared_ptr<SecretTen> GradTensor = GetTenById(gradId);
    shared_ptr<SecretTen> MomentumTensor = (momentumId != 0) ? GetTenById(momentumId) : nullptr;

    const int total_num_elem = ParamTensor->GetNumElem();
    auto& chunk_manager = TrustedChunkManager::getInstance();
    DtypeForCpuOp *param_store_chunk, *grad_store_chunk, *momentum_store_chunk;
    ChunkGuard<DtypeForCpuOp> param_guard(StoreChunkPool::GetChunkPool(), param_store_chunk);
    ChunkGuard<DtypeForCpuOp> grad_guard(StoreChunkPool::GetChunkPool(), grad_store_chunk);
    ChunkGuard<DtypeForCpuOp> momentum_guard(StoreChunkPool::GetChunkPool(), momentum_store_chunk);

    auto store_chunk_op = [&](int start_store_chunk, int num_elem_in_store_chunk) {
        chunk_manager.GetChunk(ParamTensor->GetChunkId(start_store_chunk), param_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(GradTensor->GetChunkId(start_store_chunk), grad_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.GetChunk(MomentumTensor->GetChunkId(start_store_chunk), momentum_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));

        auto chunk_op = [&](int start, int num_elem_in_op) {
            DtypeForCpuOp* param_arr = param_store_chunk + start;
            DtypeForCpuOp* grad_arr = grad_store_chunk + start;
            DtypeForCpuOp* momentum_arr = momentum_store_chunk + start;
            if (first_momentum) {
                for(size_t j = 0; j < num_elem_in_op; j++) {
                    grad_arr[j] += weight_decay * param_arr[j];
                    momentum_arr[j] = grad_arr[j];
                    param_arr[j] -= lr * momentum_arr[j];
                }
            } else {
                for(size_t j = 0; j < num_elem_in_op; j++) {
                    grad_arr[j] += weight_decay * param_arr[j];
                    momentum_arr[j] = momentum_arr[j] * momentum + (1 - dampening) * grad_arr[j];
                    param_arr[j] -= lr * momentum_arr[j];
                }
            }
        };
        run_all_chunks(chunk_op, WORK_CHUNK_ELEM, num_elem_in_store_chunk);
        chunk_manager.StoreChunk(ParamTensor->GetChunkId(start_store_chunk), param_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
        chunk_manager.StoreChunk(MomentumTensor->GetChunkId(start_store_chunk), momentum_store_chunk, num_elem_in_store_chunk * sizeof(DtypeForCpuOp));
    };
    run_all_chunks(store_chunk_op, STORE_CHUNK_ELEM, total_num_elem);
}

void SecretStochasticQuantize(IdT src_id, IdT dst_id, uint64_t q_tag) {
    quantize_stochastic(GetTenById(src_id), GetTenById(dst_id), q_tag);
}

void SecretFusedQuantizeShare(IdT af_id, void* e_arr, uint64_t q_tag, uint64_t u_seed) {
    FusedQuantizeShare(GetTenById(af_id), (DtypeForCpuOp*) e_arr, q_tag, u_seed);
}

void SecretFusedQuantizeShare2(IdT af_id, void* a1_arr, void* e_arr,
        uint64_t q_tag, uint64_t a0_seed, uint64_t u_seed) {
    FusedQuantizeShare2(GetTenById(af_id), (DtypeForCpuOp*) a1_arr, (DtypeForCpuOp*) e_arr,
            q_tag, a0_seed, u_seed);
}

void SecretFusedRecon(IdT cf_id, IdT cq_id, DtypeForCpuOp* c_left_arr, uint64_t x_tag, uint64_t y_tag) {
    FusedRecon(GetTenById(cf_id), GetTenById(cq_id), (DtypeForCpuOp*) c_left_arr, x_tag, y_tag);
}

} // End of extern C
