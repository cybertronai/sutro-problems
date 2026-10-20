"""Small project-local NVRTC/CUDA driver loader. No PyTorch extension build.

Compilation must happen during ordinary warmup, before graph capture. Modules
retain their cubin and function handles for the lifetime of captured graphs.
"""
import ctypes as ct
import ctypes.util
import glob
import time


def _library(name, candidates):
    found=ctypes.util.find_library(name)
    paths=([found] if found else [])+candidates
    errors=[]
    for path in paths:
        try:return ct.CDLL(path)
        except OSError as error:errors.append(str(error))
    raise RuntimeError(f'Cannot load {name}: '+ '; '.join(errors))


class Module:
    def __init__(self, source, name='project_kernel.cu', architecture='sm_80', *, line_info=False):
        import torch
        torch.cuda.init()
        self.cuda=_library('cuda',['libcuda.so.1'])
        candidates=glob.glob('/usr/local/cuda*/targets/x86_64-linux/lib/libnvrtc.so*')
        candidates+=glob.glob('/usr/local/lib/python*/site-packages/nvidia/*/lib/libnvrtc.so*')
        self.nvrtc=_library('nvrtc',candidates)
        nv=self.nvrtc
        nv.nvrtcCreateProgram.argtypes=[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_void_p]
        nv.nvrtcCompileProgram.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)]
        for suffix in ('ProgramLog','CUBIN'):
            getattr(nv,'nvrtcGet'+suffix+'Size').argtypes=[ct.c_void_p,ct.POINTER(ct.c_size_t)]
            getattr(nv,'nvrtcGet'+suffix).argtypes=[ct.c_void_p,ct.c_void_p]
        nv.nvrtcDestroyProgram.argtypes=[ct.POINTER(ct.c_void_p)]
        program=ct.c_void_p()
        self._nvcheck(nv.nvrtcCreateProgram(ct.byref(program),source.encode(),name.encode(),0,None,None),'create')
        options=[('--gpu-architecture='+architecture).encode(),b'--std=c++17',b'--fmad=false',b'--include-path=/usr/local/cuda/include']
        if line_info:options.append(b'--generate-line-info')
        args=(ct.c_char_p*len(options))(*options)
        start=time.perf_counter()
        code=nv.nvrtcCompileProgram(program,len(options),args)
        self.compile_ms=(time.perf_counter()-start)*1000
        size=ct.c_size_t();nv.nvrtcGetProgramLogSize(program,ct.byref(size))
        log=ct.create_string_buffer(size.value);nv.nvrtcGetProgramLog(program,log)
        self.log=log.value.decode(errors='replace')
        try:
            self._nvcheck(code,'compile: '+self.log[-4000:])
            self._nvcheck(nv.nvrtcGetCUBINSize(program,ct.byref(size)),'cubin size')
            binary=ct.create_string_buffer(size.value)
            self._nvcheck(nv.nvrtcGetCUBIN(program,binary),'get cubin')
            self.binary=binary.raw
        finally:nv.nvrtcDestroyProgram(ct.byref(program))
        cu=self.cuda
        cu.cuGetErrorString.argtypes=[ct.c_int,ct.POINTER(ct.c_char_p)]
        cu.cuModuleLoadData.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p]
        cu.cuModuleGetFunction.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p]
        cu.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[ct.POINTER(ct.c_int),ct.c_void_p,ct.c_int,ct.c_size_t]
        common=[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p,ct.POINTER(ct.c_void_p)]
        cu.cuLaunchKernel.argtypes=common+[ct.c_void_p]
        cu.cuLaunchCooperativeKernel.argtypes=common
        self.handle=ct.c_void_p();self._binary=ct.create_string_buffer(self.binary)
        start=time.perf_counter()
        self._check(cu.cuModuleLoadData(ct.byref(self.handle),self._binary),'load cubin')
        self.load_ms=(time.perf_counter()-start)*1000
        self.functions={}

    def _nvcheck(self, code, operation):
        if code:raise RuntimeError(f'NVRTC {operation}: status {code}')

    def _check(self, code, operation):
        if code:
            message=ct.c_char_p();self.cuda.cuGetErrorString(code,ct.byref(message))
            raise RuntimeError(f'CUDA {operation}: {code} {message.value!r}')

    def function(self, name):
        if name not in self.functions:
            handle=ct.c_void_p()
            self._check(self.cuda.cuModuleGetFunction(ct.byref(handle),self.handle,name.encode()),'get '+name)
            self.functions[name]=handle
        return self.functions[name]

    def configure_shared(self, name, byte_count):
        function=self.function(name)
        self.cuda.cuFuncSetAttribute.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
        self._check(self.cuda.cuFuncSetAttribute(function,8,byte_count),'dynamic shared memory')

    def resident_capacity(self, name, block=128, shared=0):
        import torch
        blocks=ct.c_int()
        self._check(self.cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ct.byref(blocks),self.function(name),block,shared),'occupancy')
        return blocks.value*torch.cuda.get_device_properties(0).multi_processor_count

    def launch(self, name, grid, block, arguments, *, shared=0, cooperative=False):
        import torch
        # Tensor pointers are passed as c_void_p; scalar callers use explicit
        # ctypes types. Keep every holder alive until the launch API returns.
        holders=[ct.c_void_p(a.data_ptr()) if isinstance(a,torch.Tensor) else a for a in arguments]
        parameters=(ct.c_void_p*len(holders))(*(ct.cast(ct.byref(a),ct.c_void_p) for a in holders))
        args=(self.function(name),grid,1,1,block,1,1,shared,ct.c_void_p(torch.cuda.current_stream().cuda_stream),parameters)
        if cooperative:
            # Capacity is checked before graph capture by the caller.
            code=self.cuda.cuLaunchCooperativeKernel(*args)
        else:code=self.cuda.cuLaunchKernel(*args,None)
        self._check(code,'launch '+name)
