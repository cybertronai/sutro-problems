import triton
import triton.language as tl

@triton.jit
def normalize_kernel(X, Q, NX, NQ, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    x = tl.load(X+off,off<5400,other=0.)
    q = tl.load(Q+off,off<5400,other=0.)
    x = x*4.; x = x-0.5
    q = q*4.; q = q-0.5
    tl.store(NX+off,x,off<5400)
    tl.store(NQ+off,q,off<5400)

@triton.jit
def initialize_kernel(INITIAL, P, Y, TARGET, BLOCK: tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    initial=tl.load(INITIAL+off,off<650,other=0.)
    tl.store(P+off,initial,off<650)
    label=tl.load(Y+off//10,off<6000,other=0)
    target=(label==off%10).to(tl.float32)
    tl.store(TARGET+off,target,off<6000)

@triton.jit(do_not_specialize=['batch_start'])
def hidden_kernel(X,P,H,batch_start,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=off//32; hidden=off%32; valid=off<960
    acc=tl.full((BLOCK,),0.,tl.float32)
    for f in tl.static_range(9):
        x=tl.load(X+(batch_start+row)*9+f,valid,other=0.)
        w=tl.load(P+f*32+hidden,valid,other=0.)
        product=x*w
        acc=acc+product
    acc=acc+tl.load(P+288+hidden,valid,other=0.)
    h=tl.where(acc>0.,acc,0.)
    tl.store(H+off,h,valid)

@triton.jit(do_not_specialize=['batch_start'])
def delta2_kernel(H,P,TARGET,D2,batch_start,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=off//10; c=off%10; valid=off<300
    acc=tl.full((BLOCK,),0.,tl.float32)
    for h in tl.static_range(32):
        a=tl.load(H+row*32+h,valid,other=0.)
        w=tl.load(P+320+h*10+c,valid,other=0.)
        product=a*w
        acc=acc+product
    acc=acc+tl.load(P+640+c,valid,other=0.)
    target=tl.load(TARGET+(batch_start+row)*10+c,valid,other=0.)
    tl.store(D2+off,acc-target,valid)

@triton.jit
def delta1_kernel(H,P,D2,D1,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=off//32; h=off%32; valid=off<960
    acc=tl.full((BLOCK,),0.,tl.float32)
    for c in tl.static_range(10):
        d=tl.load(D2+row*10+c,valid,other=0.)
        w=tl.load(P+320+h*10+c,valid,other=0.)
        product=d*w
        acc=acc+product
    active=tl.load(H+off,valid,other=0.)>0.
    tl.store(D1+off,tl.where(active,acc,0.),valid)

@triton.jit(do_not_specialize=['batch_start'])
def update_kernel(X,H,D1,D2,P,batch_start,STEP:tl.constexpr,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    w1=off<288
    b1=(off>=288)&(off<320)
    w2=(off>=320)&(off<640)
    b2=(off>=640)&(off<650)
    hidden1=off%32
    hidden2=(off-320)//10
    class2=(off-320)%10
    grad=tl.full((BLOCK,),0.,tl.float32)
    for row in tl.static_range(30):
        x=tl.load(X+(batch_start+row)*9+off//32,w1,other=0.)
        d1=tl.load(D1+row*32+hidden1,w1|b1,other=0.)
        h=tl.load(H+row*32+hidden2,w2,other=0.)
        d2=tl.load(D2+row*10+class2,w2,other=0.)
        d2bias=tl.load(D2+row*10+(off-640),b2,other=0.)
        term=tl.where(w1,x*d1,tl.where(b1,d1,tl.where(w2,h*d2,d2bias)))
        grad=grad+term
    change=STEP*grad
    old=tl.load(P+off,off<650,other=0.)
    tl.store(P+off,old-change,off<650)

@triton.jit
def inference_hidden_kernel(Q,P,H,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=off//32; hidden=off%32; valid=off<19200
    acc=tl.full((BLOCK,),0.,tl.float32)
    for f in tl.static_range(9):
        q=tl.load(Q+row*9+f,valid,other=0.)
        w=tl.load(P+f*32+hidden,valid,other=0.)
        product=q*w
        acc=acc+product
    acc=acc+tl.load(P+288+hidden,valid,other=0.)
    tl.store(H+off,tl.where(acc>0.,acc,0.),valid)

@triton.jit
def inference_output_kernel(H,P,SCORES,OUT,BLOCK:tl.constexpr):
    rows=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=rows<600
    best=tl.full((BLOCK,),float('-inf'),tl.float32)
    winner=tl.full((BLOCK,),0,tl.int32)
    for c in tl.static_range(10):
        acc=tl.full((BLOCK,),0.,tl.float32)
        for h in tl.static_range(32):
            x=tl.load(H+rows*32+h,valid,other=0.)
            w=tl.load(P+320+h*10+c)
            product=x*w
            acc=acc+product
        acc=acc+tl.load(P+640+c)
        tl.store(SCORES+rows*10+c,acc,valid)
        change=acc>best
        winner=tl.where(change,c,winner)
        best=tl.where(change,acc,best)
    tl.store(OUT+rows,winner,valid)
@triton.jit(do_not_specialize=['batch_start'])
def hidden_cached(X, XC, P, H, batch_start, PANELS: tl.constexpr):
    off = tl.arange(0, 512)
    v = tl.load(X + batch_start * 9 + off, off < 270, other=0.)
    tl.store(XC + off, v, off < 270)
    tl.debug_barrier()
    if PANELS:
        rows = tl.arange(0, 32)
        for panel in tl.static_range(8):
            cols = panel * 4 + tl.arange(0, 4)
            acc = tl.full((32, 4), 0., tl.float32)
            for f in tl.static_range(9):
                a = tl.load(XC + rows * 9 + f, rows < 30, other=0.)
                b = tl.load(P + f * 32 + cols)
                acc = acc + a[:, None] * b[None, :]
            acc = acc + tl.load(P + 288 + cols)[None, :]
            tl.store(H + rows[:, None] * 32 + cols[None, :],
                     tl.where(acc > 0., acc, 0.), rows[:, None] < 30)
    else:
        idx = tl.arange(0, 1024)
        row = idx // 32
        col = idx % 32
        acc = tl.full((1024,), 0., tl.float32)
        for f in tl.static_range(9):
            a = tl.load(XC + row * 9 + f, idx < 960, other=0.)
            b = tl.load(P + f * 32 + col, idx < 960, other=0.)
            acc = acc + a * b
        acc = acc + tl.load(P + 288 + col, idx < 960, other=0.)
        tl.store(H + idx, tl.where(acc > 0., acc, 0.), idx < 960)

@triton.jit(do_not_specialize=['batch_start'])
def output_panel(H, P, TARGET, D2, batch_start):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 16)
    acc = tl.full((8, 16), 0., tl.float32)
    for k in tl.static_range(32):
        a = tl.load(H + rows * 32 + k, rows < 30, other=0.)
        b = tl.load(P + 320 + k * 10 + cols, cols < 10, other=0.)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 640 + cols, cols < 10, other=0.)[None, :]
    valid = (rows[:, None] < 30) & (cols[None, :] < 10)
    target = tl.load(TARGET + (batch_start + rows[:, None]) * 10 + cols[None, :], valid, other=0.)
    tl.store(D2 + rows[:, None] * 10 + cols[None, :], acc - target, valid)

@triton.jit
def backward_panel(H, P, D2, D1):
    rows = tl.arange(0, 32)
    hidden = tl.program_id(0) * 4 + tl.arange(0, 4)
    acc = tl.full((32, 4), 0., tl.float32)
    for k in tl.static_range(10):
        a = tl.load(D2 + rows * 10 + k, rows < 30, other=0.)
        b = tl.load(P + 320 + hidden * 10 + k)
        acc = acc + a[:, None] * b[None, :]
    addresses = rows[:, None] * 32 + hidden[None, :]
    active = tl.load(H + addresses, rows[:, None] < 30, other=0.) > 0.
    tl.store(D1 + addresses, tl.where(active, acc, 0.), rows[:, None] < 30)

@triton.jit
def update_panels(XC, H, D1, D2, P, STEP: tl.constexpr):
    panel = tl.program_id(0)
    if panel < 8:
        features = tl.arange(0, 16)
        hidden = panel * 4 + tl.arange(0, 4)
        grad = tl.full((16, 4), 0., tl.float32)
        bias = tl.full((4,), 0., tl.float32)
        for row in tl.static_range(30):
            a = tl.load(XC + row * 9 + features, features < 9, other=0.)
            b = tl.load(D1 + row * 32 + hidden)
            grad = grad + a[:, None] * b[None, :]
            bias = bias + b
        address = features[:, None] * 32 + hidden[None, :]
        old = tl.load(P + address, features[:, None] < 9, other=0.)
        tl.store(P + address, old - STEP * grad, features[:, None] < 9)
        old_bias = tl.load(P + 288 + hidden)
        tl.store(P + 288 + hidden, old_bias - STEP * bias)
    else:
        hidden2 = (panel - 8) * 8 + tl.arange(0, 8)
        cols2 = tl.arange(0, 16)
        grad2 = tl.full((8, 16), 0., tl.float32)
        bias2 = tl.full((16,), 0., tl.float32)
        for row2 in tl.static_range(30):
            a2 = tl.load(H + row2 * 32 + hidden2)
            b2 = tl.load(D2 + row2 * 10 + cols2, cols2 < 10, other=0.)
            grad2 = grad2 + a2[:, None] * b2[None, :]
            if panel == 8:
                bias2 = bias2 + b2
        address2 = 320 + hidden2[:, None] * 10 + cols2[None, :]
        old2 = tl.load(P + address2, cols2[None, :] < 10, other=0.)
        tl.store(P + address2, old2 - STEP * grad2, cols2[None, :] < 10)
        if panel == 8:
            old_bias2 = tl.load(P + 640 + cols2, cols2 < 10, other=0.)
            tl.store(P + 640 + cols2, old_bias2 - STEP * bias2, cols2 < 10)

@triton.jit
def inference_hidden_panels(Q, P, H):
    rows = tl.program_id(0) * 30 + tl.arange(0, 32)
    valid = tl.arange(0, 32) < 30
    # Retain each query feature in registers across the eight column panels.
    q0 = tl.load(Q + rows * 9 + 0, valid, other=0.)
    q1 = tl.load(Q + rows * 9 + 1, valid, other=0.)
    q2 = tl.load(Q + rows * 9 + 2, valid, other=0.)
    q3 = tl.load(Q + rows * 9 + 3, valid, other=0.)
    q4 = tl.load(Q + rows * 9 + 4, valid, other=0.)
    q5 = tl.load(Q + rows * 9 + 5, valid, other=0.)
    q6 = tl.load(Q + rows * 9 + 6, valid, other=0.)
    q7 = tl.load(Q + rows * 9 + 7, valid, other=0.)
    q8 = tl.load(Q + rows * 9 + 8, valid, other=0.)
    for panel in tl.static_range(8):
        cols = panel * 4 + tl.arange(0, 4)
        acc = tl.full((32, 4), 0., tl.float32)
        b0 = tl.load(P + 0 * 32 + cols)
        acc = acc + q0[:, None] * b0[None, :]
        b1 = tl.load(P + 1 * 32 + cols)
        acc = acc + q1[:, None] * b1[None, :]
        b2 = tl.load(P + 2 * 32 + cols)
        acc = acc + q2[:, None] * b2[None, :]
        b3 = tl.load(P + 3 * 32 + cols)
        acc = acc + q3[:, None] * b3[None, :]
        b4 = tl.load(P + 4 * 32 + cols)
        acc = acc + q4[:, None] * b4[None, :]
        b5 = tl.load(P + 5 * 32 + cols)
        acc = acc + q5[:, None] * b5[None, :]
        b6 = tl.load(P + 6 * 32 + cols)
        acc = acc + q6[:, None] * b6[None, :]
        b7 = tl.load(P + 7 * 32 + cols)
        acc = acc + q7[:, None] * b7[None, :]
        b8 = tl.load(P + 8 * 32 + cols)
        acc = acc + q8[:, None] * b8[None, :]
        acc = acc + tl.load(P + 288 + cols)[None, :]
        tl.store(H + rows[:, None] * 32 + cols[None, :], tl.where(acc > 0., acc, 0.), valid[:, None])

@triton.jit
def inference_hidden_cached(Q, P, H):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, 32)
    acc = tl.full((4, 32), 0., tl.float32)
    for f in tl.static_range(9):
        a = tl.load(Q + rows * 9 + f, rows < 600, other=0.)
        b = tl.load(P + f * 32 + cols)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 288 + cols)[None, :]
    tl.store(H + rows[:, None] * 32 + cols[None, :], tl.where(acc > 0., acc, 0.), rows[:, None] < 600)

@triton.jit
def inference_output_panels(H, P, SCORES, OUT):
    rows = tl.program_id(0) * 30 + tl.arange(0, 32)
    valid_rows = tl.arange(0, 32) < 30
    cols = tl.arange(0, 16)
    acc = tl.full((32, 16), 0., tl.float32)
    for h in tl.static_range(32):
        a = tl.load(H + rows * 32 + h, valid_rows, other=0.)
        b = tl.load(P + 320 + h * 10 + cols, cols < 10, other=0.)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 640 + cols, cols < 10, other=0.)[None, :]
    tl.store(SCORES + rows[:, None] * 10 + cols[None, :], acc, valid_rows[:, None] & (cols[None, :] < 10))
    masked = tl.where(cols[None, :] < 10, acc, float('-inf'))
    best = tl.max(masked, axis=1)
    winner = tl.min(tl.where((cols[None, :] < 10) & (masked == best[:, None]), cols[None, :], 2147483647), axis=1)
    tl.store(OUT + rows, winner, valid_rows)
