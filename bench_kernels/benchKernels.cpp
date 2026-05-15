// Standalone flash-attention kernel benchmark.
// Measures pure GPU kernel time via profiling events (OCL) / timestamp queries (Vulkan).
// No MNN runtime — kernel source/SPIR-V files are read from the MNN source tree.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <map>
#include <chrono>

#ifndef OCL_KERNEL_PATH
#define OCL_KERNEL_PATH "../contrib/MNN-simiyutin/../../opencl-flash/qkvt/kernels/flash_attn_v2_mnn.cl"
#endif
#ifndef VK_SPV_DIR
#define VK_SPV_DIR "../contrib/MNN-simiyutin/source/backend/vulkan/buffer/compiler/.cache/shader/cache/"
#endif

// ── OpenCL ───────────────────────────────────────────────────────────────────
#ifdef BENCH_OCL
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif
#endif

// ── Vulkan ───────────────────────────────────────────────────────────────────
#ifdef BENCH_VK
#include <vulkan/vulkan.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────

struct BenchCase { int B, H, L, D; };

static void fillRandom(float* p, int n) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < n; i++) p[i] = dist(rng);
}

static void printResult(const char* tag, int B, int H, int L, int D,
                        float minUs, float avgUs, float maxUs) {
    double gflops = 4.0*B*H*(double)L*L*D / (minUs/1e6) / 1e9;
    printf("%-20s  B=%d H=%2d L=%5d D=%3d  "
           "min=%8.0f  avg=%8.0f  max=%8.0f us  GFLOPS=%.2f\n",
           tag, B, H, L, D, minUs, avgUs, maxUs, gflops);
}

static std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); return {}; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenCL
// ─────────────────────────────────────────────────────────────────────────────
#ifdef BENCH_OCL

static cl_device_id  g_oclDev  = nullptr;
static cl_context    g_oclCtx  = nullptr;
static cl_command_queue g_oclQ = nullptr;

static bool oclInit() {
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    if (!np) return false;
    std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np, plats.data(), nullptr);
    for (auto pl : plats) {
        cl_uint nd = 0; clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
        if (!nd) continue;
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        g_oclDev = devs[0]; break;
    }
    if (!g_oclDev) return false;
    cl_int err;
    g_oclCtx = clCreateContext(nullptr, 1, &g_oclDev, nullptr, nullptr, &err);
    g_oclQ   = clCreateCommandQueue(g_oclCtx, g_oclDev, 0, &err);
    char name[256]; clGetDeviceInfo(g_oclDev, CL_DEVICE_NAME, 256, name, nullptr);
    printf("[OCL] %s\n", name);
    return true;
}

static void benchOcl(const BenchCase& c, int warmup, int iters) {
    // Tile params matching the OCL kernel defaults
    const int BLOCK_M = 64, BLOCK_N = 32, TPR = 2;

    // OCL kernel uses 2 tiles: Q_tile[BLOCK_M*D] + KV_tile[BLOCK_N*D]
    size_t shmem = (size_t)(BLOCK_M + BLOCK_N) * c.D * sizeof(float);
    // PATCH: subtract 256-byte safety margin — NVIDIA OpenCL reports 48KB total but
    // the kernel uses a few extra bytes for bookkeeping, causing "too much shared data".
    size_t maxShmem = 0;
    clGetDeviceInfo(g_oclDev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxShmem), &maxShmem, nullptr);
    maxShmem = (maxShmem > 256) ? maxShmem - 256 : maxShmem;
    if (shmem > (size_t)maxShmem) {
        printf("ocl-flash             B=%d H=%2d L=%5d D=%3d  SKIP (shmem %zu > limit %llu bytes)\n",
               c.B, c.H, c.L, c.D, shmem, maxShmem);
        return;
    }

    auto src = readFile(OCL_KERNEL_PATH);
    if (src.empty()) return;
    const char* ptr = reinterpret_cast<const char*>(src.data());
    size_t len = src.size();

    cl_int err;
    cl_program prog = clCreateProgramWithSource(g_oclCtx, 1, &ptr, &len, &err);
    char opts[256];
    snprintf(opts, sizeof(opts), "-D D_HEAD=%d -D BLOCK_SIZE_M=%d -D BLOCK_SIZE_N=%d -D THREADS_PER_ROW=%d",
             c.D, BLOCK_M, BLOCK_N, TPR);
    err = clBuildProgram(prog, 1, &g_oclDev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(prog, g_oclDev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
        std::string log(logLen, '\0');
        clGetProgramBuildInfo(prog, g_oclDev, CL_PROGRAM_BUILD_LOG, logLen, log.data(), nullptr);
        fprintf(stderr, "[OCL] build error:\n%s\n", log.c_str());
        clReleaseProgram(prog); return;
    }
    cl_kernel k = clCreateKernel(prog, "flash_attention_v2_mnn_fwd", &err);
    clReleaseProgram(prog);

    int elems = c.B * c.H * c.L * c.D;
    size_t bytes = elems * sizeof(float);
    std::vector<float> h(elems); fillRandom(h.data(), elems);

    cl_mem dQ = clCreateBuffer(g_oclCtx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, bytes, h.data(), &err);
    cl_mem dK = clCreateBuffer(g_oclCtx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, bytes, h.data(), &err);
    cl_mem dV = clCreateBuffer(g_oclCtx, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR, bytes, h.data(), &err);
    cl_mem dO = clCreateBuffer(g_oclCtx, CL_MEM_WRITE_ONLY,                       bytes, nullptr,  &err);

    float scale = 1.f / sqrtf((float)c.D); int causal = 0;
    clSetKernelArg(k, 0, sizeof(cl_mem), &dQ); clSetKernelArg(k, 1, sizeof(cl_mem), &dK);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dV); clSetKernelArg(k, 3, sizeof(cl_mem), &dO);
    clSetKernelArg(k, 4, sizeof(int), &c.B);   clSetKernelArg(k, 5, sizeof(int), &c.H);
    clSetKernelArg(k, 6, sizeof(int), &c.L);   clSetKernelArg(k, 7, sizeof(int), &c.L);
    clSetKernelArg(k, 8, sizeof(float), &scale); clSetKernelArg(k, 9, sizeof(int), &causal);

    int wgSize = BLOCK_M * TPR;
    size_t global[3] = {(size_t)(((c.L+BLOCK_M-1)/BLOCK_M)*wgSize), (size_t)c.B, (size_t)c.H};
    size_t local[3]  = {(size_t)wgSize, 1, 1};

    for (int i = 0; i < warmup; i++) {
        clEnqueueNDRangeKernel(g_oclQ, k, 3, nullptr, global, local, 0, nullptr, nullptr);
        clFinish(g_oclQ);
    }

    std::vector<float> times;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        clEnqueueNDRangeKernel(g_oclQ, k, 3, nullptr, global, local, 0, nullptr, nullptr);
        clFinish(g_oclQ);
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<float, std::micro>(t1 - t0).count());
    }

    float minUs = *std::min_element(times.begin(), times.end());
    float maxUs = *std::max_element(times.begin(), times.end());
    float avgUs = std::accumulate(times.begin(), times.end(), 0.f) / iters;
    int numWG = (int)(global[0]/local[0]) * (int)global[1] * (int)global[2];
    printResult("ocl-flash", c.B, c.H, c.L, c.D, minUs, avgUs, maxUs);
    printf("  wg=%d  local=[%d,1,1]  global=[%zu,%zu,%zu]  shmem=%zuB  BLOCK_M=%d BLOCK_N=%d TPR=%d\n",
           numWG, wgSize, global[0], global[1], global[2], shmem, BLOCK_M, BLOCK_N, TPR);

    clReleaseMemObject(dQ); clReleaseMemObject(dK);
    clReleaseMemObject(dV); clReleaseMemObject(dO);
    clReleaseKernel(k);
}
#endif // BENCH_OCL

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan
// ─────────────────────────────────────────────────────────────────────────────
#ifdef BENCH_VK

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
    fprintf(stderr,"VK error %d at %s:%d\n",_r,__FILE__,__LINE__); return; } } while(0)

static VkInstance       g_vkInst   = VK_NULL_HANDLE;
static VkPhysicalDevice g_vkPhys   = VK_NULL_HANDLE;
static VkDevice         g_vkDev    = VK_NULL_HANDLE;
static VkQueue          g_vkQueue  = VK_NULL_HANDLE;
static VkCommandPool    g_vkCmdPool= VK_NULL_HANDLE;
static uint32_t         g_vkQFam   = 0;
static float            g_vkTsPeriod = 1.f;
static bool             g_vkHasCoopMat = false;

static bool vkInit() {
    // Query available extensions first — only enable what's present (MoltenVK portability)
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availExts.data());

    std::vector<const char*> wantExts = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    std::vector<const char*> enabledExts;
    for (auto want : wantExts)
        for (auto& e : availExts)
            if (strcmp(e.extensionName, want) == 0) { enabledExts.push_back(want); break; }

    bool hasPortability = false;
    for (auto e : enabledExts)
        if (strcmp(e, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) { hasPortability = true; break; }

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo        = &ai;
    ici.enabledExtensionCount   = (uint32_t)enabledExts.size();
    ici.ppEnabledExtensionNames = enabledExts.data();
    ici.flags = hasPortability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
    if (vkCreateInstance(&ici, nullptr, &g_vkInst) != VK_SUCCESS) {
        fprintf(stderr, "[VK] vkCreateInstance failed\n"); return false;
    }

    uint32_t n = 0; vkEnumeratePhysicalDevices(g_vkInst, &n, nullptr);
    if (!n) { fprintf(stderr, "[VK] no physical devices\n"); return false; }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g_vkInst, &n, devs.data());
    g_vkPhys = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_vkPhys, &props);
    g_vkTsPeriod = props.limits.timestampPeriod;
    printf("[VK]  %s\n", props.deviceName);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_vkPhys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vkPhys, &nq, qfp.data());
    for (uint32_t i = 0; i < nq; i++)
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_vkQFam = i; break; }

    // Check for cooperative matrix device extension
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(g_vkPhys, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(g_vkPhys, nullptr, &devExtCount, devExts.data());
    bool hasCoopMatExt = false, hasMemModelExt = false;
    for (auto& e : devExts) {
        if (strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) hasCoopMatExt = true;
        if (strcmp(e.extensionName, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) == 0) hasMemModelExt = true;
    }
    g_vkHasCoopMat = hasCoopMatExt;
    printf("[VK]  cooperative_matrix: %s\n", hasCoopMatExt ? "yes" : "no");

    std::vector<const char*> devExtNames;
    if (hasCoopMatExt) devExtNames.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (hasMemModelExt) devExtNames.push_back(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);

    // Build feature chain for coopmat + memory model
    VkPhysicalDeviceVulkanMemoryModelFeatures memModelFeatures{};
    memModelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    memModelFeatures.vulkanMemoryModel = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures{};
    coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopMatFeatures.cooperativeMatrix = VK_TRUE;
    coopMatFeatures.pNext = hasMemModelExt ? (void*)&memModelFeatures : nullptr;

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_vkQFam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExtNames.size();
    dci.ppEnabledExtensionNames = devExtNames.empty() ? nullptr : devExtNames.data();
    dci.pNext = hasCoopMatExt ? (void*)&coopMatFeatures : nullptr;
    if (vkCreateDevice(g_vkPhys, &dci, nullptr, &g_vkDev) != VK_SUCCESS) {
        fprintf(stderr, "[VK] vkCreateDevice failed\n"); return false;
    }
    vkGetDeviceQueue(g_vkDev, g_vkQFam, 0, &g_vkQueue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = g_vkQFam;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    return vkCreateCommandPool(g_vkDev, &cpci, nullptr, &g_vkCmdPool) == VK_SUCCESS;
}

static uint32_t vkMemType(uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_vkPhys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1<<i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    return 0;
}

struct VkBuf {
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
    void destroy() { vkDestroyBuffer(g_vkDev,buf,nullptr); vkFreeMemory(g_vkDev,mem,nullptr); }
};

static VkBuf vkMakeBuf(VkDeviceSize sz, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp) {
    VkBuf b;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = sz; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(g_vkDev, &bci, nullptr, &b.buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vkDev, b.buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = vkMemType(mr.memoryTypeBits, mp);
    vkAllocateMemory(g_vkDev, &mai, nullptr, &b.mem);
    vkBindBufferMemory(g_vkDev, b.buf, b.mem, 0);
    return b;
}

static void vkUpload(VkBuf& dst, const void* src, VkDeviceSize sz) {
    VkBuf stg = vkMakeBuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* p; vkMapMemory(g_vkDev, stg.mem, 0, sz, 0, &p);
    memcpy(p, src, sz); vkUnmapMemory(g_vkDev, stg.mem);
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_vkCmdPool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_vkDev, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy r{0,0,sz}; vkCmdCopyBuffer(cb, stg.buf, dst.buf, 1, &r);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    vkQueueSubmit(g_vkQueue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(g_vkQueue);
    vkFreeCommandBuffers(g_vkDev, g_vkCmdPool, 1, &cb);
    stg.destroy();
}

// Uniform block matching sdpa_flash_mnn.comp
struct SdpaParams { int batch,seq_len,kv_seq_len,num_heads,kv_num_heads,head_dim; float scale; };

struct VkTileConfig { int blockM, blockN, tpr; };

static VkTileConfig vkSelectTile(int dHead, size_t maxShmem) {
    // Try largest BLOCK_M first, fall back to smaller
    for (int blockM : {64, 32}) {
        int blockN = 32;
        int tpr = (blockM == 64) ? 2 : 4;
        size_t shmem = (size_t)(blockM + blockN) * dHead * sizeof(float);
        if (shmem <= maxShmem) return {blockM, blockN, tpr};
    }
    return {0, 0, 0}; // nothing fits
}

static std::string vkCompileSpv(int dHead, const VkTileConfig& tile) {
    static std::map<uint64_t,std::string> cache;
    uint64_t key = ((uint64_t)dHead << 32) | ((uint64_t)tile.blockM << 16) | tile.tpr;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const std::string compSrc = std::string(VK_SPV_DIR)
        + "../../../../execution/glsl/sdpa_flash_mnn.comp";
    std::string out = "/tmp/sdpa_flash_D" + std::to_string(dHead)
                    + "_M" + std::to_string(tile.blockM) + ".spv";
    std::string macro = (dHead == 64) ? "-DD_HEAD_64" : "-DD_HEAD_128";
    macro += " -DBLOCK_M=" + std::to_string(tile.blockM);
    macro += " -DBLOCK_N=" + std::to_string(tile.blockN);
    macro += " -DTHREADS_PER_ROW=" + std::to_string(tile.tpr);
    std::string cmd = std::string(
#ifdef GLSLANG_PATH
        GLSLANG_PATH
#else
        "glslangValidator"
#endif
        ) + " -V " + macro + " -o " + out + " " + compSrc + " 2>&1";
    printf("[VK] compiling D=%d BLOCK_M=%d TPR=%d shader...\n", dHead, tile.blockM, tile.tpr);
    if (system(cmd.c_str()) != 0) { fprintf(stderr,"[VK] compile failed\n"); out = ""; }
    cache[key] = out;
    return out;
}

static void benchVkFlash(const BenchCase& c, int warmup, int iters) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_vkPhys, &props);
    size_t maxShmem = props.limits.maxComputeSharedMemorySize;

    VkTileConfig tile = vkSelectTile(c.D, maxShmem);
    if (tile.blockM == 0) {
        printf("vk-flash              B=%d H=%2d L=%5d D=%3d  SKIP (no tile config fits shmem %zu bytes)\n",
               c.B, c.H, c.L, c.D, maxShmem);
        return;
    }
    size_t shmem = (size_t)(tile.blockM + tile.blockN) * c.D * sizeof(float);

    std::string spvPath = vkCompileSpv(c.D, tile);
    if (spvPath.empty()) return;
    auto spv = readFile(spvPath);
    if (spv.empty()) return;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm; vkCreateShaderModule(g_vkDev, &smci, nullptr, &sm);

    // Layout: 4 storage + 1 uniform
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (int i = 0; i < 4; i++) {
        bindings[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 5; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_vkDev, &dslci, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pipeLayout; vkCreatePipelineLayout(g_vkDev, &plci, nullptr, &pipeLayout);

    uint32_t localX = (uint32_t)(tile.blockM * tile.tpr);
    VkSpecializationMapEntry specEntry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specInfo{1, &specEntry, sizeof(uint32_t), &localX};
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &specInfo};
    cpci.layout = pipeLayout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(g_vkDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) {
        printf("[VK] pipeline create failed for D=%d, skipping\n", c.D);
        vkDestroyPipelineLayout(g_vkDev, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(g_vkDev, dsl, nullptr);
        vkDestroyShaderModule(g_vkDev, sm, nullptr);
        return;
    }

    int elems = c.B * c.H * c.L * c.D;
    VkDeviceSize bytes = elems * sizeof(float);
    std::vector<float> h(elems); fillRandom(h.data(), elems);

    VkBuf dQ = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dK = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dV = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dO = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkUpload(dQ, h.data(), bytes); vkUpload(dK, h.data(), bytes); vkUpload(dV, h.data(), bytes);

    SdpaParams params{c.B, c.L, c.L, c.H, c.H, c.D, 1.f/sqrtf((float)c.D)};
    VkBuf uBuf = vkMakeBuf(sizeof(SdpaParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* up; vkMapMemory(g_vkDev, uBuf.mem, 0, sizeof(SdpaParams), 0, &up);
    memcpy(up, &params, sizeof(SdpaParams)); vkUnmapMemory(g_vkDev, uBuf.mem);

    VkDescriptorPoolSize poolSz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=2; dpci.pPoolSizes=poolSz;
    VkDescriptorPool descPool; vkCreateDescriptorPool(g_vkDev, &dpci, nullptr, &descPool);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=descPool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_vkDev, &dsai, &ds);

    VkDescriptorBufferInfo bi[5] = {{dQ.buf,0,bytes},{dK.buf,0,bytes},{dV.buf,0,bytes},{dO.buf,0,bytes},{uBuf.buf,0,sizeof(SdpaParams)}};
    VkWriteDescriptorSet ws[5]{};
    for (int i = 0; i < 4; i++) { ws[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&bi[i]}; }
    ws[4]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,4,0,1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bi[4]};
    vkUpdateDescriptorSets(g_vkDev, 5, ws, 0, nullptr);

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType=VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount=2;
    VkQueryPool qpool; vkCreateQueryPool(g_vkDev, &qpci, nullptr, &qpool);

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=g_vkCmdPool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    vkAllocateCommandBuffers(g_vkDev, &cbai, &cb);

    uint32_t gx = ((uint32_t)c.L + tile.blockM - 1) / tile.blockM;
    auto record = [&](bool timestamp) {
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &cbbi);
        if (timestamp) vkCmdResetQueryPool(cb, qpool, 0, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &ds, 0, nullptr);
        if (timestamp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
        vkCmdDispatch(cb, gx, (uint32_t)c.H, (uint32_t)c.B);
        if (timestamp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);
        vkEndCommandBuffer(cb);
    };
    auto submit = [&]() {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        vkQueueSubmit(g_vkQueue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(g_vkQueue);
        vkResetCommandBuffer(cb, 0);
    };

    for (int i = 0; i < warmup; i++) { record(false); submit(); }

    std::vector<float> times;
    for (int i = 0; i < iters; i++) {
        record(true); submit();
        uint64_t ts[2];
        vkGetQueryPoolResults(g_vkDev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
        times.push_back((ts[1]-ts[0]) * g_vkTsPeriod / 1e3f);
    }

    float minUs = *std::min_element(times.begin(), times.end());
    float maxUs = *std::max_element(times.begin(), times.end());
    float avgUs = std::accumulate(times.begin(), times.end(), 0.f) / iters;
    int numWG = (int)gx * c.H * c.B;
    printResult("vk-flash", c.B, c.H, c.L, c.D, minUs, avgUs, maxUs);
    printf("  wg=%d  local=[%u,1,1]  dispatch=[%u,%d,%d]  shmem=%zuB  BLOCK_M=%d BLOCK_N=%d TPR=%d\n",
           numWG, localX, gx, c.H, c.B, shmem, tile.blockM, tile.blockN, tile.tpr);

    vkDestroyQueryPool(g_vkDev, qpool, nullptr);
    vkFreeCommandBuffers(g_vkDev, g_vkCmdPool, 1, &cb);
    vkDestroyDescriptorPool(g_vkDev, descPool, nullptr);
    uBuf.destroy(); dO.destroy(); dV.destroy(); dK.destroy(); dQ.destroy();
    vkDestroyPipeline(g_vkDev, pipeline, nullptr);
    vkDestroyPipelineLayout(g_vkDev, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vkDev, dsl, nullptr);
    vkDestroyShaderModule(g_vkDev, sm, nullptr);
}
static std::string vkCompileCoopMatSpv(int dHead) {
    static std::map<int,std::string> cache;
    auto it = cache.find(dHead);
    if (it != cache.end()) return it->second;

    const std::string compSrc = std::string(VK_SPV_DIR)
        + "../../../../execution/glsl/sdpa_flash_coopmat_mnn.comp";
    std::string out = "/tmp/sdpa_coopmat_D" + std::to_string(dHead) + ".spv";
    std::string macro = (dHead == 64) ? "-DD_HEAD_64" : "-DD_HEAD_128";
    std::string cmd = std::string(
#ifdef GLSLANG_PATH
        GLSLANG_PATH
#else
        "glslangValidator"
#endif
        ) + " -V --target-env vulkan1.2 " + macro + " -o " + out + " " + compSrc + " 2>&1";
    printf("[VK] compiling coopmat D=%d shader...\n", dHead);
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "[VK] coopmat compile failed\n"); out = ""; }
    cache[dHead] = out;
    return out;
}

static void benchVkCoopMat(const BenchCase& c, int warmup, int iters) {
    if (!g_vkHasCoopMat) {
        printf("vk-coopmat            B=%d H=%2d L=%5d D=%3d  SKIP (no VK_KHR_cooperative_matrix)\n",
               c.B, c.H, c.L, c.D);
        return;
    }
    if (c.D != 64 && c.D != 128) {
        printf("vk-coopmat            B=%d H=%2d L=%5d D=%3d  SKIP (D must be 64 or 128)\n",
               c.B, c.H, c.L, c.D);
        return;
    }

    // Fixed tiling: Br=Bc=32, local_size_x=32 (one subgroup)
    static constexpr int BLOCK_M   = 32;
    static constexpr uint32_t LOCAL_X = 32;

    std::string spvPath = vkCompileCoopMatSpv(c.D);
    if (spvPath.empty()) return;
    auto spv = readFile(spvPath);
    if (spv.empty()) return;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm; vkCreateShaderModule(g_vkDev, &smci, nullptr, &sm);

    // Same layout as vk-flash: 4 storage + 1 uniform
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (int i = 0; i < 4; i++)
        bindings[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 5; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_vkDev, &dslci, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pipeLayout; vkCreatePipelineLayout(g_vkDev, &plci, nullptr, &pipeLayout);

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cpci.layout = pipeLayout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(g_vkDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) {
        printf("[VK] coopmat pipeline create failed for D=%d, skipping\n", c.D);
        vkDestroyPipelineLayout(g_vkDev, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(g_vkDev, dsl, nullptr);
        vkDestroyShaderModule(g_vkDev, sm, nullptr);
        return;
    }

    int elems = c.B * c.H * c.L * c.D;
    VkDeviceSize bytes = elems * sizeof(float);
    std::vector<float> h(elems); fillRandom(h.data(), elems);

    VkBuf dQ = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dK = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dV = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuf dO = vkMakeBuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkUpload(dQ, h.data(), bytes); vkUpload(dK, h.data(), bytes); vkUpload(dV, h.data(), bytes);

    SdpaParams params{c.B, c.L, c.L, c.H, c.H, c.D, 1.f/sqrtf((float)c.D)};
    VkBuf uBuf = vkMakeBuf(sizeof(SdpaParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* up; vkMapMemory(g_vkDev, uBuf.mem, 0, sizeof(SdpaParams), 0, &up);
    memcpy(up, &params, sizeof(SdpaParams)); vkUnmapMemory(g_vkDev, uBuf.mem);

    VkDescriptorPoolSize poolSz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=2; dpci.pPoolSizes=poolSz;
    VkDescriptorPool descPool; vkCreateDescriptorPool(g_vkDev, &dpci, nullptr, &descPool);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=descPool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_vkDev, &dsai, &ds);

    VkDescriptorBufferInfo bi[5] = {{dQ.buf,0,bytes},{dK.buf,0,bytes},{dV.buf,0,bytes},{dO.buf,0,bytes},{uBuf.buf,0,sizeof(SdpaParams)}};
    VkWriteDescriptorSet ws[5]{};
    for (int i = 0; i < 4; i++) { ws[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&bi[i]}; }
    ws[4]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,4,0,1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bi[4]};
    vkUpdateDescriptorSets(g_vkDev, 5, ws, 0, nullptr);

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType=VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount=2;
    VkQueryPool qpool; vkCreateQueryPool(g_vkDev, &qpci, nullptr, &qpool);

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=g_vkCmdPool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    vkAllocateCommandBuffers(g_vkDev, &cbai, &cb);

    // dispatch: (num_q_blocks, num_heads, batch)
    uint32_t gx = ((uint32_t)c.L + BLOCK_M - 1) / BLOCK_M;
    auto record = [&](bool timestamp) {
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &cbbi);
        if (timestamp) vkCmdResetQueryPool(cb, qpool, 0, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &ds, 0, nullptr);
        if (timestamp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
        vkCmdDispatch(cb, gx, (uint32_t)c.H, (uint32_t)c.B);
        if (timestamp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);
        vkEndCommandBuffer(cb);
    };
    auto submit = [&]() {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        vkQueueSubmit(g_vkQueue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(g_vkQueue);
        vkResetCommandBuffer(cb, 0);
    };

    for (int i = 0; i < warmup; i++) { record(false); submit(); }

    std::vector<float> times;
    for (int i = 0; i < iters; i++) {
        record(true); submit();
        uint64_t ts[2];
        vkGetQueryPoolResults(g_vkDev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
        times.push_back((ts[1]-ts[0]) * g_vkTsPeriod / 1e3f);
    }

    float minUs = *std::min_element(times.begin(), times.end());
    float maxUs = *std::max_element(times.begin(), times.end());
    float avgUs = std::accumulate(times.begin(), times.end(), 0.f) / iters;
    int numWG = (int)gx * c.H * c.B;
    printResult("vk-coopmat", c.B, c.H, c.L, c.D, minUs, avgUs, maxUs);
    printf("  wg=%d  local=[%u,1,1]  dispatch=[%u,%d,%d]  BLOCK_M=%d (coopmat Br=Bc=32 lM=lN=lK=16)\n",
           numWG, LOCAL_X, gx, c.H, c.B, BLOCK_M);

    vkDestroyQueryPool(g_vkDev, qpool, nullptr);
    vkFreeCommandBuffers(g_vkDev, g_vkCmdPool, 1, &cb);
    vkDestroyDescriptorPool(g_vkDev, descPool, nullptr);
    uBuf.destroy(); dO.destroy(); dV.destroy(); dK.destroy(); dQ.destroy();
    vkDestroyPipeline(g_vkDev, pipeline, nullptr);
    vkDestroyPipelineLayout(g_vkDev, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vkDev, dsl, nullptr);
    vkDestroyShaderModule(g_vkDev, sm, nullptr);
}
#endif // BENCH_VK

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    int warmup = 3, iters = 10;
    if (argc >= 2) warmup = atoi(argv[1]);
    if (argc >= 3) iters  = atoi(argv[2]);

    std::vector<BenchCase> cases;
    for (int seq : {1024, 2048, 4096, 8192, 16384, 32768})
        for (int d : {64, 128})
            cases.push_back({1, 4, seq, d});

#ifdef BENCH_OCL
    printf("\n=== OCL flash-attn ===\n");
    if (oclInit())
        for (auto& c : cases) benchOcl(c, warmup, iters);
#endif

#ifdef BENCH_VK
    printf("\n=== Vulkan flash-attn ===\n");
    if (vkInit()) {
        for (auto& c : cases) benchVkFlash(c, warmup, iters);
        printf("\n=== Vulkan coopmat flash-attn ===\n");
        for (auto& c : cases) benchVkCoopMat(c, warmup, iters);
    } else {
        fprintf(stderr, "[VK] init failed\n");
    }
#endif

    return 0;
}
