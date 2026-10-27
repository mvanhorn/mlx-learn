// Bounded inexact CG on an explicitly compiled, capability-checked Metal HVP.
// Included by _metal_loop.mm to share its validated DLPack/ownership machinery.

struct CGPipelines {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> initialize, product, reduce, update;
};

static std::shared_ptr<CGPipelines> cg_pipeline_for(const char* source) {
    static std::mutex mutex;
    static std::list<std::string> order;
    static std::unordered_map<std::string, std::pair<std::shared_ptr<CGPipelines>, std::list<std::string>::iterator>> cache;
    std::lock_guard<std::mutex> guard(mutex);
    auto found = cache.find(source);
    if (found != cache.end()) {
        order.splice(order.begin(), order, found->second.second);
        return found->second.first;
    }
    auto pl = std::make_shared<CGPipelines>();
    pl->device = MTLCreateSystemDefaultDevice();
    if (!pl->device) throw std::runtime_error("no Metal device for CG");
    pl->queue = [pl->device newCommandQueue];
    if (!pl->queue) throw std::runtime_error("Metal CG queue creation failed");
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = NO;
    id<MTLLibrary> lib = [pl->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
    if (!lib) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal CG compilation failed");
    auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> f = [lib newFunctionWithName:name];
        NSError* error = nil;
        id<MTLComputePipelineState> p = f ? [pl->device newComputePipelineStateWithFunction:f error:&error] : nil;
        if (!p) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal CG pipeline failed");
        if (p.staticThreadgroupMemoryLength > pl->device.maxThreadgroupMemoryLength)
            throw std::runtime_error("Metal CG pipeline exceeds threadgroup memory");
        return p;
    };
    pl->initialize = make(@"cg_initialize"); pl->product = make(@"cg_product");
    pl->reduce = make(@"cg_reduce"); pl->update = make(@"cg_update");
    order.emplace_front(source);
    cache.emplace(order.front(), std::make_pair(pl, order.begin()));
    if (cache.size() > 16) { cache.erase(order.back()); order.pop_back(); }
    return pl;
}

static PyObject* run_cg(PyObject* module, PyObject* args) {
    const char* source; PyObject* capsules; PyObject* owner_bundle;
    int n, d, p, blocks, groups, max_iter;
    double forcing_value;
    if (!PyArg_ParseTuple(args, "sOiiiiiidO", &source, &capsules, &n, &d, &p,
                         &blocks, &groups, &max_iter, &forcing_value, &owner_bundle)) return nullptr;
    if (!PyTuple_Check(capsules) || PyTuple_Size(capsules) != 5 ||
        !PyTuple_Check(owner_bundle) || PyTuple_Size(owner_bundle) != 6 ||
        PyTuple_GetItem(owner_bundle, 0) != capsules || n <= 0 || d <= 0 || d > 4096 ||
        p != d + 1 || blocks <= 0 || (groups != 2 && groups != 4 && groups != 8) ||
        max_iter < 1 || max_iter > 512 || !std::isfinite(forcing_value) ||
        forcing_value < 1e-4 || forcing_value > 0.5 || size_t(blocks)*p*4 > 64*1024*1024) {
        PyErr_SetString(PyExc_ValueError, "invalid native CG arguments"); return nullptr;
    }
    int expected_groups=d<=1024 ? (n>=16384 ? 4 : 2) : 8;
    int features=d<=1024 ? 1 : 4;
    int rows=n>=16384 ? (d<=2048 ? 32 : 64) : (d<=1024 ? 16 : 8);
    int expected_blocks=int((int64_t(n)+(expected_groups/features)*rows-1)/((expected_groups/features)*rows));
    if (groups!=expected_groups || blocks!=expected_blocks) {
        PyErr_SetString(PyExc_ValueError,"CG launch geometry mismatch"); return nullptr;
    }
    // The private source and launch specialization must agree before encoding.
    for (auto pair : {std::make_pair("N",n), {"D",d}, {"P",p}, {"B",blocks}, {"GROUPS",groups}, {"FEATURE_GROUPS",features}, {"ROWS",rows}}) {
        std::string define = std::string("#define ") + pair.first + " " + std::to_string(pair.second) + "\n";
        if (!std::strstr(source, define.c_str())) {
            PyErr_SetString(PyExc_ValueError, "CG source specialization mismatch"); return nullptr;
        }
    }
    @autoreleasepool {
      @try {
       try {
        auto registry = module_registry(module);
        reap_retained_work(registry);
        ExternalBuffer X=tensor(capsules,0,{n,d}), curvature=tensor(capsules,1,{n});
        ExternalBuffer gradient=tensor(capsules,2,{p}), reg=tensor(capsules,3,{p}), output=tensor(capsules,4,{p});
        auto pl = cg_pipeline_for(source);
        if (NSUInteger(groups)*32 > pl->product.maxTotalThreadsPerThreadgroup ||
            NSUInteger(groups)*32 > pl->device.maxThreadsPerThreadgroup.width ||
            pl->device.maxThreadsPerThreadgroup.width < 128 ||
            pl->initialize.maxTotalThreadsPerThreadgroup < 128 ||
            pl->reduce.maxTotalThreadsPerThreadgroup < 128 || pl->update.maxTotalThreadsPerThreadgroup < 128)
            throw std::runtime_error("CG pipeline threadgroup size unsupported");
        for (auto input : {X, curvature, gradient, reg, output})
            if (input.buffer.device != pl->device) throw std::runtime_error("CG buffers belong to another Metal device");
        for (auto input : {X, curvature, gradient, reg}) {
            // Conservative: private output is always a distinct MLX allocation.
            if (input.buffer == output.buffer)
                throw std::runtime_error("CG output must not alias inputs");
        }
        size_t workspace = size_t(blocks)*p*4 + size_t(p)*5*4 + 64;
        if (workspace > pl->device.maxBufferLength) throw std::runtime_error("CG workspace exceeds device limit");
        auto parts = create_buffer(pl->device,size_t(blocks)*p*4);
        auto solution = create_buffer(pl->device,size_t(p)*4);
        auto residual = create_buffer(pl->device,size_t(p)*4);
        auto direction = create_buffer(pl->device,size_t(p)*4);
        auto product = create_buffer(pl->device,size_t(p)*4);
        auto meta = create_buffer(pl->device,32);
        float forcing = float(forcing_value);
        std::string failure;
        int encoded = 0;
        WorkGuard owners(registry,owner_bundle);
        {
        ReleaseGIL released;
        id<MTLCommandBuffer> pending = nil;
        @try {
          while (encoded < max_iter) {
            id<MTLCommandBuffer> cb = [pl->queue commandBuffer];
            if (!cb) { failure="CG command buffer creation failed"; break; }
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            if (!e) { failure="CG compute encoder creation failed"; break; }
            if (encoded == 0) {
                bind(e,gradient.buffer,0,gradient.offset); bind(e,solution,1); bind(e,residual,2);
                bind(e,direction,3); bind(e,meta,4); [e setBytes:&forcing length:4 atIndex:5];
                dispatch(e,pl->initialize,128);
            }
            int count = std::min(8,max_iter-encoded);
            for (int i=0;i<count;++i) {
                bind(e,X.buffer,0,X.offset); bind(e,curvature.buffer,1,curvature.offset);
                bind(e,direction,2); bind(e,parts,3); bind(e,meta,4);
                dispatch(e,pl->product,size_t(blocks)*groups*32,groups*32);
                bind(e,parts,0); bind(e,direction,1); bind(e,reg.buffer,2,reg.offset);
                bind(e,product,3); bind(e,meta,4);
                dispatch(e,pl->reduce,((p+127)/128)*128);
                bind(e,solution,0); bind(e,residual,1); bind(e,direction,2); bind(e,product,3); bind(e,meta,4);
                dispatch(e,pl->update,128);
            }
            [e endEncoding];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            if (!blit) { failure="CG output encoder creation failed"; break; }
            [blit copyFromBuffer:solution sourceOffset:0 toBuffer:output.buffer destinationOffset:output.offset size:size_t(p)*4];
            [blit endEncoding];
            pending=cb; set_pending(registry,owners.work,cb);
            [cb commit]; [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                failure=cb.error.localizedDescription.UTF8String ?: "CG command failed"; break;
            }
            set_pending(registry,owners.work,nil); pending=nil;
            encoded += count;
            if (static_cast<float*>(meta.contents)[0] != 0) break;
          }
        } @catch (NSException* exception) {
            failure=exception.reason.UTF8String ?: "CG Metal exception";
            try_finish_pending(pending,failure);
        }
        }
        if (!failure.empty()) throw std::runtime_error(failure);
        auto* h=static_cast<float*>(meta.contents);
        return Py_BuildValue("iiddd",int(h[1]),int(h[0]),double(h[2]),double(h[3]),double(h[4]));
       } catch (const std::exception& e) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError,e.what()); return nullptr;
       }
      } @catch (NSException* exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.reason.UTF8String ?: "CG Metal API exception"); return nullptr;
      }
    }
}
