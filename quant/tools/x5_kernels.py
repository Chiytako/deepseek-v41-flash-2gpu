# X5: SM120 で参照 tilelang カーネルが動くかを個別に試す
import sys, traceback, torch
sys.path.insert(0, "/ref")
p = torch.cuda.get_device_properties(0)
print("dev", p.name, "cc", p.major, p.minor, "smem_optin", getattr(p, "shared_memory_per_block_optin", None), "sms", p.multi_processor_count)
print("torch", torch.__version__)
import tilelang; print("tilelang", tilelang.__version__)
import kernel as K
torch.manual_seed(0)
def run(name, f):
    try:
        f(); torch.cuda.synchronize(); print("OK  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "")
x = torch.randn(256, 5120, device="cuda", dtype=torch.bfloat16)
def t_act():
    y, s = K.act_quant(x, 128); print("   act_quant", y.dtype, s.shape)
run("act_quant", t_act)
def t_fp4q():
    y, s = K.fp4_act_quant(x, 32); print("   fp4_act_quant", y.dtype, y.shape, s.dtype)
run("fp4_act_quant", t_fp4q)
import inspect
print(inspect.signature(K.fp8_gemm), inspect.signature(K.fp4_gemm), inspect.signature(K.sparse_attn), inspect.signature(K.hc_split_sinkhorn))

# 参照実装の既定 (model.py の set_dtype): fp8_block_size=32? scale_fmt=ue8m0 -> 実際の値を見る
import model as M
print("fp8_block", M.fp8_block_size if hasattr(M,'fp8_block_size') else None, "fp4_block", getattr(M,'fp4_block_size',None), "scale_fmt", getattr(M,'scale_fmt',None), "scale_dtype", getattr(M,'scale_dtype',None))
torch.set_default_dtype(torch.bfloat16)
sd = torch.float8_e8m0fnu
def ref_dq_fp8(w, s, bs=32):
    return (w.float().unflatten(0,(-1,bs)).unflatten(-1,(-1,bs)) * s.float()[:,None,:,None]).flatten(2).flatten(0,1)
def t_fp8():
    N,Kd=2048,5120
    w=(torch.randn(N,Kd,device="cuda")*0.05).to(torch.float8_e4m3fn); s=torch.full((N//32,Kd//32),1.0,device="cuda").to(sd)
    a,as_=K.act_quant(x,32,"ue8m0",sd) if 'scale_fmt' in inspect.signature(K.act_quant).parameters else K.act_quant(x,32)
    c=K.fp8_gemm(a,as_,w,s,sd,block_size=32)
    ref=x.float()@ref_dq_fp8(w,s).T
    print("   fp8 cos", torch.nn.functional.cosine_similarity(c.float().flatten(),ref.flatten(),0).item())
run("fp8_gemm", t_fp8)
def t_fp4():
    N,Kd=2304,5120
    raw=torch.randint(0,256,(N,Kd//2),device="cuda",dtype=torch.uint8)
    w=raw.view(torch.float4_e2m1fn_x2); s=torch.full((N,Kd//32),1.0,device="cuda").to(sd)
    a,as_=K.act_quant(x,32,"ue8m0",sd) if 'scale_fmt' in inspect.signature(K.act_quant).parameters else K.act_quant(x,32)
    c=K.fp4_gemm(a,as_,w,s,sd,act_block_size=32)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device="cuda",dtype=torch.float32)
    wf=torch.stack([lut[(raw&15).long()],lut[(raw>>4).long()]],-1).flatten(1)
    ref=x.float()@wf.float().T
    print("   fp4 cos", torch.nn.functional.cosine_similarity(c.float().flatten(),ref.flatten(),0).item())
run("fp4_gemm", t_fp4)
def t_attn():
    b,s,h,d,T=1,64,64,512,4096
    q=torch.randn(b,s,h,d,device="cuda")*0.1; kv=torch.randn(b,T,d,device="cuda")*0.1
    sink=torch.zeros(h,device="cuda",dtype=torch.float32); idx=torch.randint(0,T,(b,s,640),device="cuda",dtype=torch.int32)
    o=K.sparse_attn(q,kv,sink,idx,d**-0.5); print("   attn", o.shape)
run("sparse_attn h64", t_attn)
def t_attn32():
    b,s,h,d,T=1,64,32,512,4096
    q=torch.randn(b,s,h,d,device="cuda")*0.1; kv=torch.randn(b,T,d,device="cuda")*0.1
    sink=torch.zeros(h,device="cuda",dtype=torch.float32); idx=torch.randint(0,T,(b,s,640),device="cuda",dtype=torch.int32)
    o=K.sparse_attn(q,kv,sink,idx,d**-0.5); print("   attn", o.shape)
run("sparse_attn h32(TP2)", t_attn32)
def t_hc():
    mixes=torch.randn(1,64,24,device="cuda",dtype=torch.float32); sc=torch.ones(3,device="cuda",dtype=torch.float32); base=torch.zeros(24,device="cuda",dtype=torch.float32)
    pre,post,comb=K.hc_split_sinkhorn(mixes,sc,base); print("   hc", comb.shape, comb[0,0].sum(-1))
run("hc_split_sinkhorn", t_hc)
