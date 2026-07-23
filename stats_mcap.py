#!/usr/bin/env python3
import struct, sys, os, json, subprocess
class R:
    def __init__(s,b): s.b=b; s.i=0
    def u16(s): v=struct.unpack_from("<H",s.b,s.i)[0]; s.i+=2; return v
    def u32(s): v=struct.unpack_from("<I",s.b,s.i)[0]; s.i+=4; return v
    def u64(s): v=struct.unpack_from("<Q",s.b,s.i)[0]; s.i+=8; return v
    def s_(s): n=s.u32(); v=s.b[s.i:s.i+n].decode("utf-8","replace"); s.i+=n; return v
def records(buf):
    i,n=0,len(buf)
    while i+9<=n:
        op=buf[i]; length=struct.unpack_from("<Q",buf,i+1)[0]
        st=i+9; en=st+length
        if en>n: break
        yield op,buf[st:en]; i=en
def zd(d): return subprocess.run(["zstd","-d","-c"],input=d,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout

def upd(stats, key, val):
    if not isinstance(val,(int,float)) or isinstance(val,bool): return
    s=stats.setdefault(key,[float("inf"),float("-inf"),0,0.0])
    s[0]=min(s[0],val); s[1]=max(s[1],val); s[2]+=1; s[3]+=val

def main(path, nchunks=45):
    size=os.path.getsize(path)
    with open(path,"rb") as f:
        f.seek(size-8-29); foot=f.read(8+29)[:29]
        ss,so,crc=struct.unpack_from("<QQI",foot,9)
        f.seek(ss); summary=f.read((size-8-29)-ss)
        channels={}; chunk_offsets=[]
        for op,c in records(summary):
            r=R(c)
            if op==0x04:
                cid=r.u16(); r.u16(); topic=r.s_(); menc=r.s_(); channels[cid]=(topic,menc)
            elif op==0x08:
                r.u64();r.u64();cso=r.u64();cl=r.u64();chunk_offsets.append((cso,cl))
        chunk_offsets.sort()
        n=len(chunk_offsets)
        pick=sorted(set(int(i*(n-1)/(nchunks-1)) for i in range(nchunks)))
        # per-topic numeric stats + bool-true tracking
        numstats={t:{} for t,_ in channels.values()}
        booltrue={t:{} for t,_ in channels.values()}
        msgcount={t:0 for t,_ in channels.values()}
        def collect(prefix,obj,topic):
            if isinstance(obj,dict):
                for k,v in obj.items(): collect(f"{prefix}{k}.",v,topic)
            elif isinstance(obj,list):
                for idx,v in enumerate(obj): collect(f"{prefix}{idx}.",v,topic)
            elif isinstance(obj,bool):
                if obj: booltrue[topic][prefix[:-1]]=booltrue[topic].get(prefix[:-1],0)+1
            elif isinstance(obj,(int,float)):
                upd(numstats[topic],prefix[:-1],obj)
        for ci in pick:
            cso,cl=chunk_offsets[ci]
            f.seek(cso); rec=f.read(cl); r=R(rec[9:])
            r.u64();r.u64();r.u64();r.u32(); comp=r.s_(); rl=r.u64()
            raw=zd(r.b[r.i:r.i+rl])
            for op,c in records(raw):
                if op!=0x05: continue
                cid=struct.unpack_from("<H",c,0)[0]
                if cid not in channels: continue
                topic,menc=channels[cid]
                msgcount[topic]+=1
                if menc=="json":
                    try: obj=json.loads(c[22:].decode("utf-8","replace"))
                    except: continue
                    collect("",obj,topic)
        print(f"Sampled {len(pick)} of {n} chunks (spread across full recording)\n")
        for topic in numstats:
            print("="*70); print(topic, f"  [{msgcount[topic]:,} msgs sampled]"); print("="*70)
            for k,(mn,mx,cnt,tot) in sorted(numstats[topic].items()):
                if cnt==0: continue
                avg=tot/cnt
                if mn==mx: print(f"  {k:32s} const = {mn:.4g}")
                else:      print(f"  {k:32s} min={mn:.4g}  max={mx:.4g}  avg={avg:.4g}")
            bt=booltrue[topic]
            if bt:
                print("  -- boolean flags observed TRUE at least once --")
                for k,cnt in sorted(bt.items()): print(f"     {k} (x{cnt})")
            print()

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "20260615_180456_T4_bulk.mcap")
