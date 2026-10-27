// Optional native command-buffer loop for wide binary logistic regression.
// Evaluated Metal DLPack capsules remain owned until submitted work reaches a
// terminal state, including if Metal's wait or status APIs throw exceptions.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

struct DLDevice { int device_type, device_id; };
struct DLDataType { uint8_t code, bits; uint16_t lanes; };
struct DLTensor {
    void* data; DLDevice device; int ndim; DLDataType dtype;
    int64_t* shape; int64_t* strides; uint64_t byte_offset;
};
struct DLManagedTensor { DLTensor dl_tensor; void* manager_ctx; void (*deleter)(DLManagedTensor*); };

struct ReleaseGIL {
    PyThreadState* state;
    ReleaseGIL() : state(PyEval_SaveThread()) {}
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

// DLPack owners are Python objects. Metal completion handlers must not call
// Python because they can execute while the interpreter is finalizing. Keep
// owners in a registry instead: normal calls reap immediately under the GIL;
// an uncertain exceptional submission is reaped by a subsequent call once its
// command buffer reports a terminal status. If a device never reaches a
// terminal state, keeping the allocation alive is safer than reusing it while
// GPU work may still be reading it.
struct RetainedWork {
    PyObject* owners = nullptr;  // One reference owned by the registry.
    id<MTLCommandBuffer> pending = nil;
    bool active = true;
};

struct RetentionRegistry {
    std::mutex mutex;
    std::vector<std::shared_ptr<RetainedWork>> works;
};

// Each module instance (and therefore each interpreter lifetime) gets a
// separate registry. An unresolved entry from a destroyed interpreter must
// never be decref'd by a later interpreter, even if CPython reuses its ID.
struct ModuleState {
    std::shared_ptr<RetentionRegistry>* registry;
};

static std::shared_ptr<RetentionRegistry> module_registry(PyObject* module) {
    auto* state = static_cast<ModuleState*>(PyModule_GetState(module));
    if (!state || !state->registry) throw std::runtime_error("Metal retention registry unavailable");
    return *state->registry;
}

// Called only with the GIL. Do not DECREF under retained_mutex: releasing a
// capsule may invoke arbitrary Python destructors that re-enter the library.
static void reap_retained_work(const std::shared_ptr<RetentionRegistry>& registry) {
    for (;;) {
        PyObject* release = nullptr;
        {
            std::lock_guard<std::mutex> guard(registry->mutex);
            auto it = registry->works.begin();
            while (it != registry->works.end()) {
                auto& work = *it;
                bool done = !work->active && !work->pending;
                if (!work->active && work->pending) {
                    @try {
                        MTLCommandBufferStatus status = work->pending.status;
                        done = status == MTLCommandBufferStatusCompleted ||
                               status == MTLCommandBufferStatusError;
                    } @catch (NSException* exception) {
                        (void)exception;
                        // An unobservable status is not proof of completion.
                    }
                }
                if (done) {
                    release = work->owners;
                    work->owners = nullptr;
                    registry->works.erase(it);
                    break;
                }
                ++it;
            }
        }
        if (!release) return;
        Py_DECREF(release);
    }
}

static std::shared_ptr<RetainedWork> retain_work(
    const std::shared_ptr<RetentionRegistry>& registry, PyObject* owners) {
    auto work = std::make_shared<RetainedWork>();
    work->owners = owners;
    Py_INCREF(owners);
    try {
        std::lock_guard<std::mutex> guard(registry->mutex);
        registry->works.push_back(work);
    } catch (...) {
        Py_DECREF(owners);
        throw;
    }
    return work;
}

static void set_pending(const std::shared_ptr<RetentionRegistry>& registry,
                        const std::shared_ptr<RetainedWork>& work, id<MTLCommandBuffer> pending) {
    std::lock_guard<std::mutex> guard(registry->mutex);
    work->pending = pending;
}

static void finish_work(const std::shared_ptr<RetentionRegistry>& registry,
                        const std::shared_ptr<RetainedWork>& work) {
    {
        std::lock_guard<std::mutex> guard(registry->mutex);
        work->active = false;
    }
    reap_retained_work(registry);
}

struct WorkGuard {
    std::shared_ptr<RetentionRegistry> registry;
    std::shared_ptr<RetainedWork> work;
    WorkGuard(std::shared_ptr<RetentionRegistry> r, PyObject* owners)
        : registry(std::move(r)), work(retain_work(registry, owners)) {}
    ~WorkGuard() noexcept {
        // ReleaseGIL has already been destroyed. If a locking failure prevents
        // cleanup, leave owners retained rather than releasing uncertain work.
        try { finish_work(registry,work); } catch (...) {}
    }
};

static void try_finish_pending(id<MTLCommandBuffer> pending, std::string& failure) {
    @try {
        if (pending && pending.status >= MTLCommandBufferStatusCommitted &&
            pending.status < MTLCommandBufferStatusCompleted)
            [pending waitUntilCompleted];
    } @catch (NSException* wait_exception) {
        (void)wait_exception;
        failure += "; failed to confirm completion of the submitted Metal command buffer";
    }
}

// Small fake command buffer for exercising the same retention registry and
// exceptional-wait path without requiring a failing GPU or unsafe fault
// injection in the Metal driver. These private helpers are used by tests.
@interface MLXLearnRetainTestBufferV7 : NSObject {
@public
    std::atomic<int> current_status;
    std::atomic<int> status_failures;
    std::atomic<int> wait_failures;
}
@end

@implementation MLXLearnRetainTestBufferV7
- (MTLCommandBufferStatus)status {
    if (status_failures.load() > 0) {
        status_failures.fetch_sub(1);
        @throw [NSException exceptionWithName:@"TestStatusFailure" reason:@"simulated status failure" userInfo:nil];
    }
    return static_cast<MTLCommandBufferStatus>(current_status.load());
}
- (void)waitUntilCompleted {
    if (wait_failures.load() > 0) {
        wait_failures.fetch_sub(1);
        @throw [NSException exceptionWithName:@"TestWaitFailure" reason:@"simulated wait failure" userInfo:nil];
    }
}
@end

struct RetainTestToken {
    std::shared_ptr<RetentionRegistry> registry;
    std::shared_ptr<RetainedWork> work;
    MLXLearnRetainTestBufferV7* buffer;
};

static RetainTestToken* test_token(PyObject* capsule) {
    return static_cast<RetainTestToken*>(PyCapsule_GetPointer(capsule, "mlxlearn.retention-test"));
}

static void delete_test_token(PyObject* capsule) {
    RetainTestToken* token = test_token(capsule);
    if (token) delete token;
    else PyErr_Clear();
}

static PyObject* test_begin(PyObject* module, PyObject* owners) {
    @autoreleasepool {
      @try {
        try {
            auto registry = module_registry(module);
            reap_retained_work(registry);
            auto work = retain_work(registry,owners);
            MLXLearnRetainTestBufferV7* fake = [[MLXLearnRetainTestBufferV7 alloc] init];
            if (!fake) { finish_work(registry,work); throw std::runtime_error("test buffer allocation failed"); }
            fake->current_status.store(MTLCommandBufferStatusCommitted);
            fake->status_failures.store(0);
            fake->wait_failures.store(0);
            set_pending(registry,work, (id<MTLCommandBuffer>)fake);
            auto* token = new RetainTestToken{registry,work, fake};
            PyObject* capsule = PyCapsule_New(token, "mlxlearn.retention-test", delete_test_token);
            if (!capsule) {
                set_pending(registry,work,nil);
                finish_work(registry,work);
                delete token;
            }
            return capsule;
        } catch (const std::exception& e) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError,e.what());
            return nullptr;
        }
      } @catch (NSException* e) {
        PyErr_SetString(PyExc_RuntimeError,e.reason.UTF8String ?: "test retention exception");
        return nullptr;
      }
    }
}

static PyObject* test_fail_waits(PyObject*, PyObject* args) {
    PyObject* capsule; int wait_errors, status_errors;
    if (!PyArg_ParseTuple(args,"Oii",&capsule,&wait_errors,&status_errors)) return nullptr;
    RetainTestToken* token = test_token(capsule);
    if (!token) return nullptr;
    token->buffer->wait_failures.store(wait_errors);
    token->buffer->status_failures.store(status_errors);
    std::string failure;
    @try {
        [token->buffer waitUntilCompleted];
    } @catch (NSException* exception) {
        (void)exception;
        failure = "simulated first wait failure";
        try_finish_pending((id<MTLCommandBuffer>)token->buffer, failure);
    }
    try { finish_work(token->registry,token->work); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); return nullptr; }
    return PyUnicode_FromString(failure.c_str());
}

static PyObject* test_set_terminal(PyObject*, PyObject* capsule) {
    RetainTestToken* token = test_token(capsule);
    if (!token) return nullptr;
    token->buffer->current_status.store(MTLCommandBufferStatusCompleted);
    token->buffer->status_failures.store(0);
    // A wheel built with Python 3.13 headers can run on Python 3.10/3.11,
    // where None is not immortal. Py_RETURN_NONE in 3.13 assumes immortality.
    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject* reap_completed(PyObject* module, PyObject*) {
    try { reap_retained_work(module_registry(module)); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); return nullptr; }
    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject* test_retained_count(PyObject* module, PyObject*) {
    try {
        auto registry = module_registry(module);
        std::lock_guard<std::mutex> guard(registry->mutex);
        return PyLong_FromSize_t(registry->works.size());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError,e.what()); return nullptr;
    }
}

struct ExternalBuffer { id<MTLBuffer> buffer; NSUInteger offset; };

static ExternalBuffer tensor(PyObject* capsules, Py_ssize_t index, const std::vector<int64_t>& shape) {
    PyObject* capsule = PyTuple_GetItem(capsules, index);
    if (!capsule) throw std::runtime_error("missing Metal DLPack capsule");
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (!managed) throw std::runtime_error("invalid Metal DLPack capsule");
    const DLTensor& t = managed->dl_tensor;
    if (t.device.device_type != 8 || t.device.device_id != 0 || t.dtype.code != 2 ||
        t.dtype.bits != 32 || t.dtype.lanes != 1 || t.ndim != int(shape.size()) || !t.data)
        throw std::runtime_error("native loop requires float32 Metal DLPack tensors");
    int64_t size = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.shape[i] != shape[i] || (t.strides && t.strides[i] != size))
            throw std::runtime_error("native loop requires contiguous Metal tensors");
        size *= t.shape[i];
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)t.data;
    if (![buffer conformsToProtocol:@protocol(MTLBuffer)] || t.byte_offset % 4 ||
        t.byte_offset > buffer.length || uint64_t(size) * 4 > buffer.length - t.byte_offset)
        throw std::runtime_error("invalid Metal buffer extent");
    return {buffer, NSUInteger(t.byte_offset)};
}

static id<MTLBuffer> create_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> result = [device newBufferWithLength:std::max(bytes, size_t(4))
                                              options:MTLResourceStorageModeShared];
    if (!result || !result.contents) throw std::runtime_error("Metal buffer allocation failed");
    memset(result.contents, 0, result.length);
    return result;
}

struct Pipelines {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> direction, speculative, reduce, accept, active, backward, finish;
};

static std::shared_ptr<Pipelines> pipeline_for(const char* source) {
    static std::mutex lock;
    static std::list<std::string> order;
    static std::unordered_map<std::string, std::pair<std::shared_ptr<Pipelines>, std::list<std::string>::iterator>> cache;
    std::lock_guard<std::mutex> guard(lock);
    auto found=cache.find(source);
    if(found!=cache.end()) {
        order.splice(order.begin(),order,found->second.second);
        return found->second.first;
    }
    auto pl=std::make_shared<Pipelines>();
    pl->device=MTLCreateSystemDefaultDevice();
    if(!pl->device) throw std::runtime_error("no Metal device");
    pl->queue=[pl->device newCommandQueue];
    if(!pl->queue) throw std::runtime_error("Metal command queue creation failed");
    NSError* error=nil;
    id<MTLLibrary> lib=[pl->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
    if(!lib) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal compilation failed");
    auto make=[&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> f=[lib newFunctionWithName:name];
        NSError* pipeline_error=nil;
        id<MTLComputePipelineState> pipeline=f ? [pl->device newComputePipelineStateWithFunction:f error:&pipeline_error] : nil;
        if(!pipeline) throw std::runtime_error(pipeline_error.localizedDescription.UTF8String ?: "Metal pipeline creation failed");
        return pipeline;
    };
    pl->direction=make(@"direction_kernel"); pl->speculative=make(@"speculative_kernel");
    pl->reduce=make(@"reduce_spec"); pl->accept=make(@"accept_kernel");
    pl->active=make(@"make_active"); pl->backward=make(@"backward_kernel"); pl->finish=make(@"finish_grad");
    order.emplace_front(source);
    cache.emplace(order.front(),std::make_pair(pl,order.begin()));
    if(cache.size()>32){ cache.erase(order.back());order.pop_back(); }
    return pl;
}

struct Slice {
    id<MTLBuffer> buffer; NSUInteger offset;
    void* contents() const { return static_cast<char*>(buffer.contents)+offset; }
};

static void bind(id<MTLComputeCommandEncoder> e, Slice s, NSUInteger i) {
    [e setBuffer:s.buffer offset:s.offset atIndex:i];
}
static void bind(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, NSUInteger i, NSUInteger offset=0) {
    [e setBuffer:b offset:offset atIndex:i];
}
static void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSUInteger count, NSUInteger width=128) {
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static PyObject* run(PyObject* module, PyObject* args) {
    const char* source; PyObject* capsules; PyObject* owner_bundle;
    const char* center_data; Py_ssize_t center_bytes;
    PyObject* seed_capsules=Py_None; PyObject* device_buffers=Py_None;
    int seed_count=0, weighted=0;
    int n,d,p,m,blocks,groups,max_iter;
    double denom_d,tol_d,c1_d,mincurv_d;
    if (!PyArg_ParseTuple(args,"sOiiiiiiiddddOy#|OiOi",&source,&capsules,&n,&d,&p,&m,&blocks,&groups,&max_iter,&denom_d,&tol_d,&c1_d,&mincurv_d,&owner_bundle,&center_data,&center_bytes,&seed_capsules,&seed_count,&device_buffers,&weighted)) return nullptr;
    // The generated kernel and host-side final check must apply the same
    // progress condition. Source comes only from our private generator.
    const bool source_weighted=std::strstr(source,"#define WEIGHTED 1\n")!=nullptr;
    if ((weighted!=0 && weighted!=1) || source_weighted!=bool(weighted)) {
        PyErr_SetString(PyExc_ValueError,"weighted progress does not match native kernel source"); return nullptr;
    }
    // The vector recurrence may cooperate across several SIMD groups.
    // Keep the compiled loop stride and its actual launch width identical.
    int direction_threads=32;
    if (const char* marker=std::strstr(source,"#define DIRECTION_THREADS ")) {
        const char* value=marker+std::strlen("#define DIRECTION_THREADS ");
        char* end=nullptr;
        long width=std::strtol(value,&end,10);
        if (!end || *end!='\n' || (width!=32 && width!=64 && width!=128 && width!=256 && width!=512)) {
            PyErr_SetString(PyExc_ValueError,"invalid native direction threadgroup width"); return nullptr;
        }
        direction_threads=int(width);
    }
    const bool seeded=seed_capsules!=Py_None;
    const bool device_io=device_buffers!=Py_None;
    const Py_ssize_t seed_owner_index=7;
    const Py_ssize_t device_owner_index=7+(seeded?4:0);
    const Py_ssize_t expected_owners=device_owner_index+(device_io?3:0);
    if (!PyTuple_Check(capsules) || PyTuple_Size(capsules)!=6 ||
        !PyTuple_Check(owner_bundle) || PyTuple_Size(owner_bundle)!=expected_owners ||
        PyTuple_GetItem(owner_bundle,0)!=capsules || n<=0 || d<=0 || p!=d+1 ||
        m<=0 || blocks<=0 || groups<=0 || max_iter<=0 ||
        seed_count<0 || seed_count>m ||
        (seeded && (!PyTuple_Check(seed_capsules) || PyTuple_Size(seed_capsules)!=3 || seed_count==0 ||
                    PyTuple_GetItem(owner_bundle,seed_owner_index)!=seed_capsules)) ||
        (!seeded && seed_count!=0) ||
        (device_io && (!PyTuple_Check(device_buffers) || PyTuple_Size(device_buffers)!=2 || center_bytes!=0 ||
                       PyTuple_GetItem(owner_bundle,device_owner_index)!=device_buffers)) ||
        (!device_io && center_bytes!=Py_ssize_t(d)*4)) {
        PyErr_SetString(PyExc_ValueError,"invalid native command loop arguments"); return nullptr;
    }
    @autoreleasepool {
      @try {
       try {
        auto registry = module_registry(module);
        reap_retained_work(registry);
        ExternalBuffer X=tensor(capsules,0,{n,d}), y=tensor(capsules,1,{n}), sw=tensor(capsules,2,{n});
        ExternalBuffer initial=tensor(capsules,3,{p}), h0=tensor(capsules,4,{p}), reg=tensor(capsules,5,{p});
        ExternalBuffer seed_s={}, seed_y={}, seed_hc={}, center_external={}, output_external={};
        if(seeded) {
          seed_s=tensor(seed_capsules,0,{m,p});seed_y=tensor(seed_capsules,1,{m,p});
          seed_hc=tensor(seed_capsules,2,{m,2});
        }
        if(device_io) {
          center_external=tensor(device_buffers,0,{d});output_external=tensor(device_buffers,1,{p});
        }
        auto pl = pipeline_for(source);
        id<MTLDevice> device=pl->device;
        if (NSUInteger(direction_threads)>pl->direction.maxTotalThreadsPerThreadgroup ||
            NSUInteger(direction_threads)>device.maxThreadsPerThreadgroup.width ||
            pl->direction.staticThreadgroupMemoryLength>device.maxThreadgroupMemoryLength)
            throw std::runtime_error("native direction exceeds Metal pipeline limits");
        if (X.buffer.device != device || y.buffer.device != device || sw.buffer.device != device ||
            initial.buffer.device != device || h0.buffer.device != device || reg.buffer.device != device ||
            (seeded && (seed_s.buffer.device != device || seed_y.buffer.device != device || seed_hc.buffer.device != device)) ||
            (device_io && (center_external.buffer.device != device || output_external.buffer.device != device)))
            throw std::runtime_error("DLPack buffers belong to a different Metal device");
        if(device_io) {
          auto overlap=[](ExternalBuffer a, size_t as, ExternalBuffer b, size_t bs) {
            return a.buffer==b.buffer && a.offset < b.offset+bs && b.offset < a.offset+as;
          };
          if(overlap(output_external,size_t(p)*4,X,size_t(n)*d*4) ||
             overlap(output_external,size_t(p)*4,y,size_t(n)*4) ||
             overlap(output_external,size_t(p)*4,sw,size_t(n)*4) ||
             overlap(output_external,size_t(p)*4,initial,size_t(p)*4) ||
             overlap(output_external,size_t(p)*4,h0,size_t(p)*4) ||
             overlap(output_external,size_t(p)*4,reg,size_t(p)*4) ||
             overlap(output_external,size_t(p)*4,center_external,size_t(d)*4) ||
             (seeded && (overlap(output_external,size_t(p)*4,seed_s,size_t(m)*p*4) ||
                         overlap(output_external,size_t(p)*4,seed_y,size_t(m)*p*4) ||
                         overlap(output_external,size_t(p)*4,seed_hc,size_t(m)*2*4))))
            throw std::runtime_error("output buffer must not alias input buffers");
        }
        int nb=(n+255)/256;
        size_t total=2*(size_t(p)*8+size_t(m)*p*8+size_t(m)*8+48+size_t(n)*4)
            +size_t(p)*12+48+size_t(n)*8+size_t(8)*blocks*4+size_t(blocks)*p*4
            +32+size_t(nb)*p*4+size_t(d)*4+4+33*256;
        id<MTLBuffer> arena=create_buffer(device,total);
        size_t cursor=0;
        auto alloc=[&](size_t size) -> Slice {
            cursor=(cursor+255)&~size_t(255);
            if(size>total-cursor) throw std::runtime_error("Metal arena exhausted");
            Slice result={arena,cursor};cursor+=size;return result;
        };
        Slice w[2],g[2],hs[2],hy[2],hc[2],meta[2],raw[2];
        for(int i=0;i<2;++i) {
          w[i]=alloc(size_t(p)*4);g[i]=alloc(size_t(p)*4);
          hs[i]=alloc(size_t(m)*p*4);hy[i]=alloc(size_t(m)*p*4);
          hc[i]=alloc(size_t(m)*2*4);meta[i]=alloc(48);raw[i]=alloc(size_t(n)*4);
        }
        static_cast<float*>(meta[0].contents())[11]=std::log(2.0f);
        if(seeded) {
          auto* initial_meta=static_cast<float*>(meta[0].contents());
          initial_meta[2]=float(seed_count);initial_meta[3]=float(seed_count%m);
        }
        if(!device_io) {
          if (!initial.buffer.contents) throw std::runtime_error("initial gradient buffer is not CPU-visible");
          memcpy(g[0].contents(),static_cast<const char*>(initial.buffer.contents)+initial.offset,size_t(p)*4);
          memcpy(g[1].contents(),g[0].contents(),size_t(p)*4);
        }
        auto direction=alloc(size_t(p)*4), stats=alloc(48);
        auto dr=alloc(size_t(n)*4), partial=alloc(size_t(8)*blocks*4);
        auto gradparts=alloc(size_t(blocks)*p*4), diffs=alloc(32);
        auto first=alloc(size_t(p)*4), residual=alloc(size_t(n)*4);
        auto backparts=alloc(size_t(nb)*p*4), flag=alloc(4), center=alloc(size_t(d)*4);
        if(!device_io) memcpy(center.contents(),center_data,size_t(d)*4);
        float denom=float(denom_d),tol=float(tol_d),c1=float(c1_d),mincurv=float(mincurv_d),alpha=1;
        int cur=0,n_iter=0,status=0,n_fev=0;
        bool first_submission=true;
        std::string failure;
        WorkGuard owners(registry,owner_bundle);
        {
        ReleaseGIL released;
        id<MTLCommandBuffer> pending = nil;
        @try {
        while (n_iter<max_iter) {
          @autoreleasepool {
            id<MTLCommandBuffer> cb=[pl->queue commandBuffer];
            if (!cb) { failure="Metal command buffer creation failed"; break; }
            if(first_submission && (seeded || device_io)) {
              id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
              if(!blit) { failure="Metal initialization blit encoder creation failed"; break; }
              if(seeded) {
                [blit copyFromBuffer:seed_s.buffer sourceOffset:seed_s.offset toBuffer:hs[0].buffer destinationOffset:hs[0].offset size:size_t(m)*p*4];
                [blit copyFromBuffer:seed_y.buffer sourceOffset:seed_y.offset toBuffer:hy[0].buffer destinationOffset:hy[0].offset size:size_t(m)*p*4];
                [blit copyFromBuffer:seed_hc.buffer sourceOffset:seed_hc.offset toBuffer:hc[0].buffer destinationOffset:hc[0].offset size:size_t(m)*2*4];
              }
              if(device_io) {
                [blit copyFromBuffer:initial.buffer sourceOffset:initial.offset toBuffer:g[0].buffer destinationOffset:g[0].offset size:size_t(p)*4];
                [blit copyFromBuffer:initial.buffer sourceOffset:initial.offset toBuffer:g[1].buffer destinationOffset:g[1].offset size:size_t(p)*4];
              }
              [blit endEncoding];
            }
            first_submission=false;
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            if (!e) { failure="Metal compute encoder creation failed"; break; }
            // Keep short-fit checks after eight and nine accepted steps, then
            // use four-step batches until a fit is demonstrably long-running.
            // Longer fits amortize synchronization in bounded 16-step chunks;
            // every command remains capped by the remaining iteration budget.
            int chunk=std::min(n_iter==0 ? 8 : n_iter==8 ? 1 : n_iter<32 ? 4 : 16,max_iter-n_iter);
            for(int k=0;k<chunk;++k) {
              int next=1-cur;
              bind(e,w[cur],0);bind(e,g[cur],1);bind(e,w[next],2);bind(e,g[next],3);
              bind(e,hs[cur],4);bind(e,hy[cur],5);bind(e,hc[cur],6);bind(e,meta[cur],7);
              bind(e,reg.buffer,8,reg.offset);[e setBytes:&tol length:4 atIndex:9];bind(e,h0.buffer,10,h0.offset);
              [e setBytes:&mincurv length:4 atIndex:11];[e setBytes:&alpha length:4 atIndex:12];
              if(device_io) bind(e,center_external.buffer,13,center_external.offset); else bind(e,center,13);bind(e,direction,14);bind(e,hs[next],15);bind(e,hy[next],16);bind(e,hc[next],17);bind(e,stats,18);
              dispatch(e,pl->direction,direction_threads,direction_threads);
              bind(e,X.buffer,0,X.offset);bind(e,direction,1);bind(e,raw[cur],2);bind(e,y.buffer,3,y.offset);
              bind(e,sw.buffer,4,sw.offset);bind(e,stats,5);[e setBytes:&denom length:4 atIndex:6];
              bind(e,dr,7);bind(e,partial,8);bind(e,gradparts,9);
              dispatch(e,pl->speculative,NSUInteger(blocks)*groups*32,groups*32);
              bind(e,partial,0);bind(e,gradparts,1);[e setBytes:&denom length:4 atIndex:2];bind(e,diffs,3);bind(e,first,4);
              dispatch(e,pl->reduce,NSUInteger((p+31)/32)*128,128);
              bind(e,w[cur],0);bind(e,raw[cur],1);bind(e,direction,2);bind(e,dr,3);bind(e,y.buffer,4,y.offset);
              bind(e,sw.buffer,5,sw.offset);bind(e,stats,6);bind(e,diffs,7);[e setBytes:&c1 length:4 atIndex:8];
              bind(e,w[next],9);bind(e,raw[next],10);bind(e,residual,11);bind(e,meta[next],12);
              dispatch(e,pl->accept,std::max(n,p),256);
              bind(e,meta[next],0);[e setBytes:&alpha length:4 atIndex:1];bind(e,flag,2);
              dispatch(e,pl->active,1,1);
              bind(e,X.buffer,0,X.offset);bind(e,residual,1);bind(e,flag,2);bind(e,backparts,3);
              [e setComputePipelineState:pl->backward];
              [e dispatchThreadgroups:MTLSizeMake((p+31)/32,nb,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
              [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
              bind(e,backparts,0);bind(e,first,1);bind(e,meta[next],2);[e setBytes:&alpha length:4 atIndex:3];
              bind(e,g[cur],4);bind(e,w[next],5);bind(e,reg.buffer,6,reg.offset);
              [e setBytes:&denom length:4 atIndex:7];bind(e,g[next],8);dispatch(e,pl->finish,p);
              cur=next;
            }
            // Publish before commit: even if commit throws, it is not safe to
            // assume the driver failed to submit the buffer.
            [e endEncoding];pending=cb;set_pending(registry,owners.work,cb);[cb commit];[cb waitUntilCompleted];
            if(cb.status != MTLCommandBufferStatusCompleted) {
              failure = cb.error.localizedDescription.UTF8String ?: "Metal command buffer failed"; break;
            }
            set_pending(registry,owners.work,nil);pending=nil;
            float* h=static_cast<float*>(meta[cur].contents());
            n_iter=int(h[0]);status=int(h[1]);n_fev=int(h[9]);
            if(!device_io) {
              float maxgrad=0;const float* grad=static_cast<const float*>(g[cur].contents());
              const float* shift=static_cast<const float*>(center.contents());
              for(int i=0;i<p;++i) {
                float original=i<d ? std::fma(shift[i],grad[d],grad[i]) : grad[i];
                if(!std::isfinite(original)) { maxgrad=INFINITY; break; }
                maxgrad=std::max(maxgrad,std::abs(original));
              }
              float recent=std::abs(h[10]);
              if(n_iter>=2) recent=0.5f*(recent+std::abs(h[4]));
              bool progress=(!weighted && (d<2048 || n_iter==0)) || maxgrad==0 ||
                  ((!weighted || n_iter>=2) && std::isfinite(h[11]) && recent<=tol*((weighted || d>=2048) ? 0.025f : 1.0f)*std::max(std::abs(h[11]),1e-12f));
              if(status==0 && maxgrad<=tol && progress) status=1;
            }
            if(status!=0) break;
          }
        }
        if(failure.empty() && device_io) {
          @autoreleasepool {
            id<MTLCommandBuffer> cb=[pl->queue commandBuffer];
            if (!cb) throw std::runtime_error("Metal output command buffer creation failed");
            id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
            if (!blit) throw std::runtime_error("Metal output blit encoder creation failed");
            [blit copyFromBuffer:w[cur].buffer sourceOffset:w[cur].offset
                        toBuffer:output_external.buffer destinationOffset:output_external.offset size:size_t(p)*4];
            [blit endEncoding];pending=cb;set_pending(registry,owners.work,cb);[cb commit];[cb waitUntilCompleted];
            if(cb.status != MTLCommandBufferStatusCompleted)
              failure=cb.error.localizedDescription.UTF8String ?: "Metal output copy failed";
            else {set_pending(registry,owners.work,nil);pending=nil;}
          }
        }
        } @catch (NSException* exception) {
          failure = exception.reason.UTF8String ?: "Metal command encoding exception";
          try_finish_pending(pending,failure);
        }
        }
        if(!failure.empty()) throw std::runtime_error(failure);
        PyObject* bytes;
        if(device_io) { Py_IncRef(Py_None); bytes=Py_None; }
        else bytes=PyBytes_FromStringAndSize(static_cast<const char*>(w[cur].contents()),Py_ssize_t(p)*4);
        if (!bytes) return nullptr;
        float* final_meta=static_cast<float*>(meta[cur].contents());
        PyObject* result=Py_BuildValue("Niiidd",bytes,n_iter,status,n_fev,double(final_meta[10]),double(final_meta[4]));
        return result;
       } catch(const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError,e.what());
        return nullptr;
       }
      } @catch (NSException* exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.reason.UTF8String ?: "Metal API exception");
        return nullptr;
      }
    }
}

#include "_metal_cg.h"

static PyMethodDef methods[]={
    {"run",run,METH_VARARGS,"Execute a bounded Metal L-BFGS loop."},
    {"run_cg",run_cg,METH_VARARGS,"Execute a bounded inexact Metal CG solve."},
    {"_reap_completed",reap_completed,METH_NOARGS,"Release owners of terminal command buffers."},
    {"_test_begin_retention",test_begin,METH_O,"Create a simulated pending submission."},
    {"_test_fail_waits",test_fail_waits,METH_VARARGS,"Simulate wait and status failures."},
    {"_test_set_terminal",test_set_terminal,METH_O,"Complete a simulated submission."},
    {"_test_retained_count",test_retained_count,METH_NOARGS,"Count retained submissions."},
    {nullptr,nullptr,0,nullptr}
};
static void free_module(void* instance) {
    auto* state = static_cast<ModuleState*>(PyModule_GetState(static_cast<PyObject*>(instance)));
    if (!state) return;
    auto* registry = state->registry;
    state->registry = nullptr;
    if (!registry) return;
    bool has_unresolved = true;
    try {
        std::lock_guard<std::mutex> guard((*registry)->mutex);
        has_unresolved = !(*registry)->works.empty();
    } catch (...) {
        // If state cannot be inspected, preserve it without allocating memory.
    }
    // Deliberately leak the whole registry if a submission is unresolved.
    // This retains both command buffers and all DLPack owners without invoking
    // Python at teardown. A future interpreter cannot reach this registry.
    if (!has_unresolved) delete registry;
}

// Multi-phase initialization is required for import in Python 3.12's
// subinterpreters. Each module instance has its own registry, including when
// a subinterpreter is destroyed and another is created in the same process.
// Py_mod_exec and PyModuleDef_Init are part of the Python 3.10 limited ABI.
static int exec_module(PyObject* instance) {
    try {
        auto* state = static_cast<ModuleState*>(PyModule_GetState(instance));
        if (!state) throw std::runtime_error("cannot initialize Metal retention registry");
        if (!state->registry)
            state->registry = new std::shared_ptr<RetentionRegistry>(std::make_shared<RetentionRegistry>());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError,e.what());
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot legacy_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(exec_module)},
    {0, nullptr}
};
// Python 3.12 isolated subinterpreters have their own GIL. The extension's
// owner registries are per-module and its process-wide pipeline caches are
// protected by mutexes, so it supports those interpreters. The new slot and
// its value are ABI constants (3 and 2), but are hidden from headers compiled
// with the Python 3.10 limited API. Do not pass the new slot to Python 3.10 or
// 3.11, which reject unrecognized slots.
static PyModuleDef_Slot isolated_slots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(exec_module)},
    {3, reinterpret_cast<void*>(2)},  // Py_mod_multiple_interpreters, per-interpreter GIL
    {0, nullptr}
};
static struct PyModuleDef legacy_module={PyModuleDef_HEAD_INIT,"_metal_loop",nullptr,sizeof(ModuleState),methods,legacy_slots,nullptr,nullptr,free_module};
static struct PyModuleDef isolated_module={PyModuleDef_HEAD_INIT,"_metal_loop",nullptr,sizeof(ModuleState),methods,isolated_slots,nullptr,nullptr,free_module};
PyMODINIT_FUNC PyInit__metal_loop(void) {
    int major=0, minor=0;
    if (std::sscanf(Py_GetVersion(), "%d.%d", &major, &minor) != 2) {
        PyErr_SetString(PyExc_RuntimeError,"cannot determine Python runtime version");
        return nullptr;
    }
    return PyModuleDef_Init((major>3 || (major==3 && minor>=12)) ? &isolated_module : &legacy_module);
}
