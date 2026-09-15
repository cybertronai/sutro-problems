"""NR-K8 Adam w32/e100 on A100: fused kernels, bitwise validation, one session."""
import hashlib
import io
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
             'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6', 'nvidia-ml-py==12.560.30')
app = modal.App('sutro-small-adam-a100')
MODES = ('fused1', 'fused2')
NODE_COUNTS = {'fused1': 2 + 1 + 1 + 4000 + 2, 'fused2': 2 + 1 + 1 + 1 + 2}

FUSED_SOURCE = '''
@triton.jit
def normalize_kernel(X, NX, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + off, off < 9000, other=0.)
    tl.store(NX + off, v * 4.0 - 0.5, off < 9000)


@triton.jit
def copy_params_kernel(INIT, P, M, V, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(INIT + off, off < 650, other=0.)
    tl.store(P + off, v, off < 650)
    tl.store(M + off, 0., off < 650)
    tl.store(V + off, 0., off < 650)


@triton.jit
def targets_kernel(Y, TARGET, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = off // 10
    col = off % 10
    y = tl.load(Y + row, row < 1000, other=0)
    tl.store(TARGET + off, tl.where(col == y, 1.0, 0.0), off < 10000)


@triton.jit(do_not_specialize=['first0', 'base', 'count'])
def fused_task(X, P, M, V, TARGET, H, D1, D2, C1, C2, BC, first0, base, count, STEP: tl.constexpr):
    rows = tl.arange(0, 32)
    feats = tl.arange(0, 16)
    hid = tl.arange(0, 32)
    cls = tl.arange(0, 16)
    for i in range(count):
        first = (first0 + i * 25) % 1000
        t = base + i
        c1 = tl.load(C1 + t)
        c2 = tl.load(C2 + t)
        bm1 = tl.load(BC + 0)
        om1 = tl.load(BC + 1)
        bm2 = tl.load(BC + 2)
        om2 = tl.load(BC + 3)
        half = tl.load(BC + 4)
        seed = tl.load(BC + 5)
        eps = tl.load(BC + 6)
        acc = tl.full((32, 32), 0., tl.float32)
        for f in tl.static_range(9):
            a = tl.load(X + (first + rows) * 9 + f, rows < 25, other=0.)
            b = tl.load(P + f * 32 + hid)
            acc = acc + a[:, None] * b[None, :]
        acc = acc + tl.load(P + 288 + hid)[None, :]
        tl.store(H + rows[:, None] * 32 + hid[None, :], tl.where(acc > 0., acc, 0.), rows[:, None] < 25)
        tl.debug_barrier()
        acc2 = tl.full((32, 16), 0., tl.float32)
        for k in tl.static_range(32):
            a2 = tl.load(H + rows * 32 + k, rows < 25, other=0.)
            b2 = tl.load(P + 320 + k * 10 + cls, cls < 10, other=0.)
            acc2 = acc2 + a2[:, None] * b2[None, :]
        acc2 = acc2 + tl.load(P + 640 + cls, cls < 10, other=0.)[None, :]
        valid2 = (rows[:, None] < 25) & (cls[None, :] < 10)
        tgt = tl.load(TARGET + (first + rows[:, None]) * 10 + cls[None, :], valid2, other=0.)
        tl.store(D2 + rows[:, None] * 10 + cls[None, :], acc2 - tgt, valid2)
        tl.debug_barrier()
        acc3 = tl.full((32, 32), 0., tl.float32)
        for k3 in tl.static_range(10):
            a3 = tl.load(D2 + rows * 10 + k3, rows < 25, other=0.)
            b3 = tl.load(P + 320 + hid * 10 + k3)
            acc3 = acc3 + a3[:, None] * b3[None, :]
        valid3 = rows[:, None] < 25
        active = tl.load(H + rows[:, None] * 32 + hid[None, :], valid3, other=0.) > 0.
        tl.store(D1 + rows[:, None] * 32 + hid[None, :], tl.where(active, acc3, 0.), valid3)
        tl.debug_barrier()
        g1 = tl.full((16, 32), 0., tl.float32)
        gb1 = tl.full((32,), 0., tl.float32)
        for r in tl.static_range(25):
            a4 = tl.load(X + (first + r) * 9 + feats, feats < 9, other=0.)
            b4 = tl.load(D1 + r * 32 + hid)
            g1 = g1 + a4[:, None] * b4[None, :]
            gb1 = gb1 + b4
        valid1 = feats[:, None] < 9
        addr1 = feats[:, None] * 32 + hid[None, :]
        m1 = tl.load(M + addr1, valid1, other=0.)
        v1 = tl.load(V + addr1, valid1, other=0.)
        p1 = tl.load(P + addr1, valid1, other=0.)
        mn = bm1 * m1 + om1 * g1
        vn = bm2 * v1 + om2 * g1 * g1
        mhat = tl.math.div_rn(mn, c1)
        vhat = tl.math.div_rn(vn, c2)
        y = vhat + seed
        for _ in tl.static_range(8):
            qq = tl.math.div_rn(vhat, y)
            qq = y + qq
            y = half * qq
        tl.store(M + addr1, mn, valid1)
        tl.store(V + addr1, vn, valid1)
        tl.store(P + addr1, p1 - tl.math.div_rn(STEP * mhat, y + eps), valid1)
        mb1 = tl.load(M + 288 + hid)
        vb1 = tl.load(V + 288 + hid)
        pb1 = tl.load(P + 288 + hid)
        mnb = bm1 * mb1 + om1 * gb1
        vnb = bm2 * vb1 + om2 * gb1 * gb1
        mhb = tl.math.div_rn(mnb, c1)
        vhb = tl.math.div_rn(vnb, c2)
        yb = vhb + seed
        for _ in tl.static_range(8):
            qb = tl.math.div_rn(vhb, yb)
            qb = yb + qb
            yb = half * qb
        tl.store(M + 288 + hid, mnb)
        tl.store(V + 288 + hid, vnb)
        tl.store(P + 288 + hid, pb1 - tl.math.div_rn(STEP * mhb, yb + eps))
        g2 = tl.full((32, 16), 0., tl.float32)
        gb2 = tl.full((16,), 0., tl.float32)
        for r2 in tl.static_range(25):
            a5 = tl.load(H + r2 * 32 + hid)
            b5 = tl.load(D2 + r2 * 10 + cls, cls < 10, other=0.)
            g2 = g2 + a5[:, None] * b5[None, :]
            gb2 = gb2 + b5
        valid5 = cls[None, :] < 10
        addr2 = 320 + hid[:, None] * 10 + cls[None, :]
        m2 = tl.load(M + addr2, valid5, other=0.)
        v2 = tl.load(V + addr2, valid5, other=0.)
        p2 = tl.load(P + addr2, valid5, other=0.)
        mn2 = bm1 * m2 + om1 * g2
        vn2 = bm2 * v2 + om2 * g2 * g2
        mh2 = tl.math.div_rn(mn2, c1)
        vh2 = tl.math.div_rn(vn2, c2)
        y2 = vh2 + seed
        for _ in tl.static_range(8):
            q2 = tl.math.div_rn(vh2, y2)
            q2 = y2 + q2
            y2 = half * q2
        tl.store(M + addr2, mn2, valid5)
        tl.store(V + addr2, vn2, valid5)
        tl.store(P + addr2, p2 - tl.math.div_rn(STEP * mh2, y2 + eps), valid5)
        validb2 = cls < 10
        mb2 = tl.load(M + 640 + cls, validb2, other=0.)
        vb2 = tl.load(V + 640 + cls, validb2, other=0.)
        pb2 = tl.load(P + 640 + cls, validb2, other=0.)
        mnb2 = bm1 * mb2 + om1 * gb2
        vnb2 = bm2 * vb2 + om2 * gb2 * gb2
        mhb2 = tl.math.div_rn(mnb2, c1)
        vhb2 = tl.math.div_rn(vnb2, c2)
        yb2 = vhb2 + seed
        for _ in tl.static_range(8):
            qb2 = tl.math.div_rn(vhb2, yb2)
            qb2 = yb2 + qb2
            yb2 = half * qb2
        tl.store(M + 640 + cls, mnb2, validb2)
        tl.store(V + 640 + cls, vnb2, validb2)
        tl.store(P + 640 + cls, pb2 - tl.math.div_rn(STEP * mhb2, yb2 + eps), validb2)
        tl.debug_barrier()


@triton.jit
def infer_hidden(Q, P, H, LIMIT: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    hid = tl.arange(0, 32)
    acc = tl.full((32, 32), 0., tl.float32)
    for f in tl.static_range(9):
        a = tl.load(Q + rows * 9 + f, rows < LIMIT, other=0.)
        b = tl.load(P + f * 32 + hid)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 288 + hid)[None, :]
    tl.store(H + rows[:, None] * 32 + hid[None, :], tl.where(acc > 0., acc, 0.), rows[:, None] < LIMIT)


@triton.jit
def infer_output(H, P, SCORES, OUT, LIMIT: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cls = tl.arange(0, 16)
    acc = tl.full((32, 16), 0., tl.float32)
    for k in tl.static_range(32):
        a = tl.load(H + rows * 32 + k, rows < LIMIT, other=0.)
        b = tl.load(P + 320 + k * 10 + cls, cls < 10, other=0.)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 640 + cls, cls < 10, other=0.)[None, :]
    tl.store(SCORES + rows[:, None] * 10 + cls[None, :], acc, (rows[:, None] < LIMIT) & (cls[None, :] < 10))
    masked = tl.where(cls[None, :] < 10, acc, float('-inf'))
    best = tl.max(masked, axis=1)
    winner = tl.min(tl.where((cls[None, :] < 10) & (masked == best[:, None]), cls[None, :], 2147483647), axis=1)
    tl.store(OUT + rows, winner, rows < LIMIT)
'''


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=16384, timeout=3600, startup_timeout=600,
              min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2, retries=0)
def benchmark(source, payload, expected, provenance):
    import ctypes as ct
    import importlib.util
    import platform
    import statistics
    import time
    import numpy as np
    import torch
    import triton
    import pynvml as nv

    torch.set_num_threads(4)
    path = Path('/tmp/adam_kernels.py')
    path.write_text('import triton\nimport triton.language as tl\nimport triton.language.math\n\n' + source)
    spec = importlib.util.spec_from_file_location('adam_kernels', path)
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    data = np.load(io.BytesIO(payload), allow_pickle=False)
    x = torch.from_numpy(data['train_images'].copy()).cuda()
    q = torch.from_numpy(data['test_images'].copy()).cuda()
    qm = torch.from_numpy(data['mutated_queries'].copy()).cuda()
    y = torch.from_numpy(data['train_labels'].astype(np.int32).copy()).cuda()
    initial = torch.from_numpy(data['initial'].copy()).cuda()
    C1 = torch.from_numpy(data['c1'].copy()).cuda()
    C2 = torch.from_numpy(data['c2'].copy()).cuda()
    BC = torch.tensor([0.9, 0.1, 0.999, 0.001, 0.5, 1e-6, 1e-8], dtype=torch.float32, device='cuda')
    nx, nq = torch.empty_like(x), torch.empty_like(q)
    P = torch.empty(650, dtype=torch.float32, device='cuda')
    M = torch.empty(650, dtype=torch.float32, device='cuda')
    V = torch.empty(650, dtype=torch.float32, device='cuda')
    TARGET = torch.empty((1000, 10), dtype=torch.float32, device='cuda')
    H = torch.empty(1024 * 32, dtype=torch.float32, device='cuda')
    D1 = torch.empty(1024 * 32, dtype=torch.float32, device='cuda')
    D2 = torch.empty(1024 * 10, dtype=torch.float32, device='cuda')
    IH = torch.empty(1024 * 32, dtype=torch.float32, device='cuda')
    scores = torch.empty((1000, 10), dtype=torch.float32, device='cuda')
    output = torch.empty(1000, dtype=torch.int32, device='cuda')
    kw = {'num_warps': 4, 'enable_fp_fusion': False}
    step = float(np.float32(.2 / 25))

    compiled = {}
    def invoke(mode):
        compiled['normalize'] = k.normalize_kernel[(9,)](x, nx, 1024, **kw)
        compiled['normalize_q'] = k.normalize_kernel[(9,)](q, nq, 1024, **kw)
        compiled['copy_params'] = k.copy_params_kernel[(1,)](initial, P, M, V, 1024, **kw)
        compiled['targets'] = k.targets_kernel[(10,)](y, TARGET, 1024, **kw)
        if mode == 'fused1':
            for mb in range(4000):
                compiled['fused'] = k.fused_task[(1,)](nx, P, M, V, TARGET, H, D1, D2, C1, C2, BC, mb * 25, mb, 1, step, **kw)
        else:
            compiled['fused'] = k.fused_task[(1,)](nx, P, M, V, TARGET, H, D1, D2, C1, C2, BC, 0, 0, 4000, step, **kw)
        compiled['infer_hidden'] = k.infer_hidden[(32,)](nq, P, IH, 1000, **kw)
        compiled['infer_output'] = k.infer_output[(32,)](IH, P, scores, output, 1000, **kw)

    arrays = {}
    for name in provenance['expected_hashes']:
        arrays[name] = np.frombuffer(expected[name], dtype=provenance['expected_dtypes'][name]).reshape(provenance['expected_shapes'][name])

    checks = {}
    def validate(case):
        record = {}
        for name, tensor in (('params', P), ('scores', scores), ('predictions', output)):
            value = tensor.cpu().numpy()
            reference = arrays[f'{case}_{name}'].astype(value.dtype)
            equal = value.view(np.uint32) == reference.view(np.uint32)
            if name == 'predictions':
                if not equal.all():
                    bad = np.flatnonzero(~equal.ravel())[:5]
                    raise AssertionError((case, name, int(equal.sum()), value.size, bad.tolist(),
                                          value.ravel()[bad].tolist(), reference.ravel()[bad].tolist()))
                maxdiff = 0.0
            else:
                maxdiff = float(np.max(np.abs(value.astype(np.float64) - reference.astype(np.float64))))
                if maxdiff > 1e-4:
                    raise AssertionError((case, name, 'maxdiff', maxdiff))
            record[name] = {'count': value.size, 'exact_fraction': float(equal.mean()),
                            'max_abs_diff': maxdiff, 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        return record

    # debug gate: exactly one minibatch must match the CPU one-step reference
    k.normalize_kernel[(9,)](x, nx, 1024, **kw)
    k.copy_params_kernel[(1,)](initial, P, M, V, 1024, **kw)
    k.targets_kernel[(10,)](y, TARGET, 1024, **kw)
    k.fused_task[(1,)](nx, P, M, V, TARGET, H, D1, D2, C1, C2, BC, 0, 0, 1, step, **kw)
    torch.cuda.synchronize()
    got = P.cpu().numpy()
    want = arrays['step1_params'].astype(got.dtype)
    same = got.view(np.uint32) == want.view(np.uint32)
    print('step1 match', int(same.sum()), '/', got.size, flush=True)

    for mode in MODES:
        print('eager validate', mode, flush=True)
        invoke(mode); torch.cuda.synchronize()
        checks[mode + '_eager'] = validate('canonical')
    y.copy_(torch.from_numpy(((data['train_labels'] + 1) % 10).astype(np.int32).copy()))
    for mode in MODES:
        invoke(mode); torch.cuda.synchronize()
        checks[mode + '_changed_labels'] = validate('changed_labels')
    y.copy_(torch.from_numpy(data['train_labels'].astype(np.int32).copy()))
    q.copy_(qm)
    for mode in MODES:
        invoke(mode); torch.cuda.synchronize()
        checks[mode + '_changed_queries'] = validate('changed_queries')
    q.copy_(torch.from_numpy(data['test_images'].copy()))

    libraries = [str(p) for p in (Path(torch.__file__).parent.parent / 'nvidia/cuda_runtime/lib').glob('libcudart.so*')]
    libraries += ['libcudart.so.12', '/usr/local/cuda/lib64/libcudart.so.12']
    runtime = None
    for library in libraries:
        try:
            runtime = ct.CDLL(library); break
        except OSError:
            pass
    assert runtime is not None
    pointer = ct.c_void_p
    for name, args in {
            'cudaStreamBeginCapture': [pointer, ct.c_int],
            'cudaStreamEndCapture': [pointer, ct.POINTER(pointer)],
            'cudaGraphGetNodes': [pointer, ct.POINTER(pointer), ct.POINTER(ct.c_size_t)],
            'cudaGraphNodeGetType': [pointer, ct.POINTER(ct.c_int)],
            'cudaGraphInstantiateWithFlags': [ct.POINTER(pointer), pointer, ct.c_ulonglong],
            'cudaGraphLaunch': [pointer, pointer], 'cudaGraphDestroy': [pointer], 'cudaGraphExecDestroy': [pointer]}.items():
        fn = getattr(runtime, name)
        fn.argtypes, fn.restype = args, ct.c_int
    def call(name, *args):
        code = getattr(runtime, name)(*args)
        assert code == 0, (name, code)
    def stream_pointer():
        return pointer(torch.cuda.current_stream().cuda_stream)
    graphs, graph_counts = {}, {}
    def replay(mode):
        call('cudaGraphLaunch', graphs[mode], stream_pointer())
    for mode in MODES:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke(mode); torch.cuda.synchronize()
            call('cudaStreamBeginCapture', stream_pointer(), 0)
            invoke(mode)
            graph = pointer()
            call('cudaStreamEndCapture', stream_pointer(), ct.byref(graph))
        n = ct.c_size_t()
        call('cudaGraphGetNodes', graph, None, ct.byref(n))
        nodes = (pointer * n.value)()
        call('cudaGraphGetNodes', graph, nodes, ct.byref(n))
        kernel_count = 0
        for node in nodes:
            kind = ct.c_int()
            call('cudaGraphNodeGetType', node, ct.byref(kind))
            kernel_count += kind.value == 0
        assert n.value == kernel_count == NODE_COUNTS[mode], (mode, n.value, kernel_count, NODE_COUNTS[mode])
        executable = pointer()
        call('cudaGraphInstantiateWithFlags', ct.byref(executable), graph, 0)
        call('cudaGraphDestroy', graph)
        graphs[mode] = executable
        graph_counts[mode] = {'total': n.value, 'kernels': kernel_count}
        replay(mode); torch.cuda.synchronize()
        checks[mode + '_graph'] = validate('canonical')
        print('graph accepted', mode, graph_counts[mode], flush=True)

    y_orig = torch.from_numpy(data['train_labels'].astype(np.int32).copy()).cuda()
    y.copy_(torch.from_numpy(((data['train_labels'] + 1) % 10).astype(np.int32).copy()))
    for mode in MODES:
        replay(mode); torch.cuda.synchronize()
        checks[mode + '_graph_changed_labels'] = validate('changed_labels')
    y.copy_(y_orig)
    q.copy_(qm)
    for mode in MODES:
        replay(mode); torch.cuda.synchronize()
        checks[mode + '_graph_changed_queries'] = validate('changed_queries')
    q.copy_(torch.from_numpy(data['test_images'].copy()))
    for mode in MODES:
        replay(mode); torch.cuda.synchronize()
        checks[mode + '_graph_restored'] = validate('canonical')
    for mode in MODES:
        replay(mode); replay(mode); torch.cuda.synchronize()
    for mode in MODES:
        checks[mode + '_reset_hash'] = validate('canonical')
    draw_inputs = {i: (torch.from_numpy(data[f'x{i}'].copy()).cuda(),
                       torch.from_numpy(data[f'q{i}'].copy()).cuda(),
                       torch.from_numpy(data[f'y{i}'].astype(np.int32).copy()).cuda()) for i in range(11)}
    for i in range(11):
        x.copy_(draw_inputs[i][0]); q.copy_(draw_inputs[i][1]); y.copy_(draw_inputs[i][2])
        invoke('fused2'); torch.cuda.synchronize()
        got = output.cpu().numpy(); want = arrays[f'pred{i}'].astype(got.dtype)
        if not (got.view(np.uint32) == want.view(np.uint32)).all():
            raise AssertionError(('draw', i, int((got == want).sum())))
    print('11-draw GPU predictions match frozen CPU predictions', flush=True)
    x.copy_(draw_inputs[0][0]); q.copy_(draw_inputs[0][1]); y.copy_(draw_inputs[0][2])
    invoke('fused2'); torch.cuda.synchronize()
    checks['restore_draw0'] = validate('canonical')

    ptx = {name: kernel.asm['ptx'] for name, kernel in compiled.items()}
    calibration = {}
    for mode in MODES:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            replay(mode)
        b.record(); b.synchronize()
        ms = a.elapsed_time(b) / 3
        calibration[mode] = {'ms': ms, 'replays': max(1, int(np.ceil(10000 / ms)))}
    print('calibration', calibration, flush=True)

    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    def decode(value):
        return value.decode() if isinstance(value, bytes) else str(value)
    def stamp():
        a = time.perf_counter(); energy = int(nv.nvmlDeviceGetTotalEnergyConsumption(handle)); b = time.perf_counter()
        return {'energy_mj': energy, 'time_s': (a + b) / 2}
    def interval(a, b):
        seconds = b['time_s'] - a['time_s']
        energy = (b['energy_mj'] - a['energy_mj']) / 1000
        return {'duration_s': seconds, 'energy_j': energy, 'average_power_w': energy / seconds}
    def idle():
        torch.cuda.synchronize(); time.sleep(3)
        a = stamp(); time.sleep(3)
        return interval(a, stamp())
    trials = []
    orders = (MODES, MODES[::-1], MODES)
    for round_id, order in enumerate(orders, 1):
        for mode in order:
            before = idle()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first = stamp(); wall = time.perf_counter(); a.record()
            repeats = calibration[mode]['replays']
            for _ in range(repeats):
                replay(mode)
            b.record(); b.synchronize(); wall = time.perf_counter() - wall
            active = interval(first, stamp()); after = idle()
            elapsed = a.elapsed_time(b) / 1000
            assert .8 < elapsed / wall < 1.2
            power = (before['average_power_w'] + after['average_power_w']) / 2
            adjusted = active['energy_j'] - power * active['duration_s']
            trials.append({'round': round_id, 'mode': mode, 'replays': repeats,
                           'cuda_ms': elapsed * 1000 / repeats, 'wall_ms': wall * 1000 / repeats,
                           'adjusted_mj': adjusted * 1000 / repeats, 'gross_mj': active['energy_j'] * 1000 / repeats,
                           'idle_before_w': before['average_power_w'], 'idle_after_w': after['average_power_w']})
            checks[f'round{round_id}_{mode}'] = validate('canonical')
            print(json.dumps(trials[-1]), flush=True)
    nv.nvmlShutdown()
    summary = {}
    for mode in MODES:
        values = [t for t in trials if t['mode'] == mode]
        summary[mode] = {'cuda_ms_median': statistics.median(t['cuda_ms'] for t in values),
                         'adjusted_mj_median': statistics.median(t['adjusted_mj'] for t in values),
                         'adjusted_mj_mean': statistics.mean(t['adjusted_mj'] for t in values),
                         'adjusted_mj_sd': statistics.stdev(t['adjusted_mj'] for t in values),
                         'gross_mj_median': statistics.median(t['gross_mj'] for t in values)}
    result = {'hardware': {'name': torch.cuda.get_device_name(0)},
              'versions': {'torch': str(torch.__version__), 'triton': triton.__version__, 'numpy': np.__version__,
                           'driver': decode(nv.nvmlSystemGetDriverVersion()), 'platform': platform.platform()},
              'provenance': provenance, 'graphs': graph_counts, 'calibration': calibration,
              'trials': trials, 'checks': checks, 'summary': summary,
              'kernel_ptx_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in ptx.items()}}
    return result, ptx


@app.local_entrypoint()
def main():
    import numpy as np
    payload = (HERE / 'generated/adam11-payload.npz').read_bytes()
    with np.load(HERE / 'generated/adam11-expected.npz') as z:
        expected = {name: z[name].tobytes() for name in z.files}
        dtypes = {name: str(z[name].dtype) for name in z.files}
        shapes = {name: list(z[name].shape) for name in z.files}
    provenance = {'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'source_sha256': hashlib.sha256(FUSED_SOURCE.encode()).hexdigest(),
                  'payload_sha256': hashlib.sha256(payload).hexdigest(),
                  'expected_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in expected.items()},
                  'expected_dtypes': dtypes, 'expected_shapes': shapes}
    destination = HERE / 'results-adam2'
    destination.mkdir(exist_ok=True)
    assert not (destination / 'adam-results.json').exists(), 'preserve completed measurements'
    result, ptx = benchmark.remote(FUSED_SOURCE, payload, expected, provenance)
    (destination / 'adam-results.json').write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'ptx').mkdir(exist_ok=True)
    for name, text in ptx.items():
        (destination / 'ptx' / (name + '.ptx')).write_text(text)
    print(json.dumps(result['summary'], indent=2))
