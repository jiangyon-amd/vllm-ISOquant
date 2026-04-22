#!/usr/bin/env python3
"""Unified crossover: GEMM+v52+S2 vs v56b+S2. Same Stage2 — isolates Stage1 diff."""
import os,sys,math,ctypes
os.environ.setdefault("TQ_ALLOW_STALE_HIP_SO","1")
_PR=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
if _PR not in sys.path: sys.path.insert(0,_PR)
import torch
DEVICE="cuda:0"; D=128

def prof(fn,w=30,r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(r)]
    for s,e in ev: s.record();fn();e.record()
    torch.cuda.synchronize()
    t=sorted([s.elapsed_time(e)*1000 for s,e in ev]); k=max(1,len(t)//10)
    return sum(t[k:-k])/len(t[k:-k])

def main():
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import _load_hip_stage1,_load_hip_stage2
    cfg=TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc",D)
    signs=generate_wht_signs(D,seed=42).to(DEVICE)
    centroids=get_centroids(D,cfg.centroid_bits).to(DEVICE)
    H_mat=_build_hadamard(D,DEVICE)
    PiT=(signs.float().unsqueeze(1)*H_mat).contiguous()
    cf32=centroids.float().contiguous()
    mp=(centroids.float().sort()[0][:-1]+centroids.float().sort()[0][1:])/2
    fn52=_load_hip_stage1(); s2bf,_=_load_hip_stage2()
    assert fn52 and s2bf
    so=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tq_decode_v56b.so")
    lib=ctypes.CDLL(so); fn56=lib.launch_tq_decode_v56b
    fn56.argtypes=([ctypes.c_void_p]*7+[ctypes.c_int]*2+[ctypes.c_int]*3+[ctypes.c_int]+[ctypes.c_int]*3+[ctypes.c_int]*4+[ctypes.c_float]+[ctypes.c_int]*2+[ctypes.c_int]*2+[ctypes.c_void_p])
    fn56.restype=None
    Hq,Hk,BS,SP=64,8,16,32; sc=1.0/math.sqrt(D); kg=Hq//Hk
    sp=torch.cuda.current_stream(DEVICE).cuda_stream
    cfgs=[(1,512),(1,2048),(1,4096),(1,8192),(4,512),(4,2048),(4,4096),(4,8192),(8,2048),(8,4096),(12,2048),(12,4096),(16,2048),(16,4096),(20,4096),(20,8192),(32,512),(32,2048)]
    print(f"\nHq={Hq} Hk={Hk} sp={SP}")
    print(f"{'='*110}")
    print(f"  {'Config':<18s} | {'GEMM+v52+S2':>12s} | {'v56b+S2':>10s} | {'Δ':>8s} | {'winner':>7s} | {'cos':>8s} | {'grid':>8s}")
    print(f"{'='*110}")
    for B,sl in cfgs:
        nb=(sl//BS)+16
        kv=torch.zeros(nb,BS,Hk,cfg.slot_size_aligned,dtype=torch.uint8,device=DEVICE)
        fk=torch.randn(sl,Hk,D,dtype=torch.bfloat16,device=DEVICE); fv=torch.randn_like(fk)
        triton_turboquant_store(fk,fv,kv,torch.arange(sl,device=DEVICE,dtype=torch.int64),PiT,centroids,mp,mse_bits=cfg.key_mse_bits,key_packed_size=cfg.key_packed_size,value_quant_bits=cfg.effective_value_quant_bits,key_fp8=cfg.key_fp8)
        q=torch.randn(B,Hq,D,dtype=torch.bfloat16,device=DEVICE)
        bp=math.ceil(sl/BS)
        bt=torch.arange(bp,device=DEVICE,dtype=torch.int32).unsqueeze(0).expand(B,-1).contiguous()
        sls=torch.full((B,),sl,device=DEVICE,dtype=torch.int32)
        qr=torch.empty(B,Hq,D,dtype=torch.float32,device=DEVICE)
        mo=torch.empty(B,Hq,SP,D+1,dtype=torch.float32,device=DEVICE)
        o1=torch.empty(B,Hq,D,dtype=torch.bfloat16,device=DEVICE)
        mo2=torch.empty(B,Hq,SP,D+1,dtype=torch.float32,device=DEVICE)
        o2=torch.empty(B,Hq,D,dtype=torch.bfloat16,device=DEVICE)
        def rr():
            torch.mm(q.reshape(B*Hq,D).float(),PiT,out=qr.reshape(B*Hq,D))
            fn52(qr.data_ptr(),kv.data_ptr(),bt.data_ptr(),sls.data_ptr(),cf32.data_ptr(),mo.data_ptr(),qr.stride(0),qr.stride(1),kv.stride(0),kv.stride(1),kv.stride(2),bt.stride(0),mo.stride(0),mo.stride(1),mo.stride(2),Hk,BS,SP,kg,sc,1,B,Hq,ctypes.c_void_p(sp))
            s2bf(mo.data_ptr(),o1.data_ptr(),sls.data_ptr(),mo.stride(0),mo.stride(1),mo.stride(2),o1.stride(0),o1.stride(1),SP,B,Hq,ctypes.c_void_p(sp))
        def rv():
            fn56(q.data_ptr(),PiT.data_ptr(),kv.data_ptr(),bt.data_ptr(),sls.data_ptr(),cf32.data_ptr(),mo2.data_ptr(),q.stride(0),q.stride(1),kv.stride(0),kv.stride(1),kv.stride(2),bt.stride(0),mo2.stride(0),mo2.stride(1),mo2.stride(2),Hk,BS,SP,kg,sc,0,1,B,Hq,ctypes.c_void_p(sp))
            s2bf(mo2.data_ptr(),o2.data_ptr(),sls.data_ptr(),mo2.stride(0),mo2.stride(1),mo2.stride(2),o2.stride(0),o2.stride(1),SP,B,Hq,ctypes.c_void_p(sp))
        rr();rv();torch.cuda.synchronize()
        cos=torch.nn.functional.cosine_similarity(o1.float().reshape(-1).unsqueeze(0),o2.float().reshape(-1).unsqueeze(0)).item()
        tr=prof(rr);tv=prof(rv);d=tv-tr
        w="v56b" if d<-1 else ("v52" if d>1 else "tie")
        g=B*Hq*SP
        print(f"  B={B:2d} seq={sl:5d} | {tr:>8.1f} us   | {tv:>7.1f} us | {d:>+6.1f}us | {w:>7s} | {cos:>8.5f} | {g:>8d}")
    print(f"{'='*110}")
if __name__=="__main__": main()
