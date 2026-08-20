"""Tabulate D1a criterion-1 verdicts from score_speed.py output.

Usage: python3 analyze.py [d1a_scores.jsonl]
"""
import json, sys, statistics as st
from collections import defaultdict
R=[json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/d1a_scores.jsonl")]
C1=0.001; SIG=0.00024
def base(a): 
    for s in ("_r1","_r2","_r3"):
        if a.endswith(s): return a[:-3]
    return a

print("="*100)
print("OLD CARD (pre-6ee71236, AISI-4340-derived) - full 8x range, n=1, arms g1_grid_prod / p5_penalty")
print("="*100)
old=[r for r in R if r["set"]=="old"]
speeds=[25.0,12.5,6.25,3.125]
for hit in [1,5,10,17]:
    print("\n-- hit %d --" % hit)
    print("%-16s %9s %9s %9s %9s | %9s %7s %8s %s" % ("arm","25.0","12.5","6.25","3.125","spread","xC1","sigma","monotonic?"))
    for arm in ["g1_grid_prod","p5_penalty"]:
        v=[next((r["iou"] for r in old if r["hit"]==hit and r["arm"]==arm and r["speed"]==s),None) for s in speeds]
        if any(x is None for x in v): continue
        sp=max(v)-min(v)
        mono = all(v[i]>=v[i+1] for i in range(3)) or all(v[i]<=v[i+1] for i in range(3))
        print("%-16s %9.4f %9.4f %9.4f %9.4f | %9.4f %7.0fx %8.0f %s"
              %(arm,v[0],v[1],v[2],v[3],sp,sp/C1,sp/SIG,"YES monotonic" if mono else "no"))

print("\n"+"="*100)
print("NEW CARD (post-6ee71236, sourced 316L) - 25.0 vs 12.5 only, n=2")
print("="*100)
new=[r for r in R if r["set"]=="new"]
for hit in [1,5,10,17]:
    print("\n-- hit %d --" % hit)
    print("%-18s %17s %17s | %9s %7s %9s %s" % ("arm","25.0 (mean+-rep)","12.5 (mean+-rep)","speed d","xC1","rep noise","signal/noise"))
    arms=sorted({base(r["arm"]) for r in new})
    for arm in arms:
        cell={}
        for s in (25.0,12.5):
            vals=[r["iou"] for r in new if r["hit"]==hit and r["speed"]==s and base(r["arm"])==arm and r["arm"].endswith(("_r1","_r2"))]
            if vals: cell[s]=vals
        if len(cell)<2: continue
        m25,m12=st.mean(cell[25.0]),st.mean(cell[12.5])
        r25=max(cell[25.0])-min(cell[25.0]) if len(cell[25.0])>1 else 0.0
        r12=max(cell[12.5])-min(cell[12.5]) if len(cell[12.5])>1 else 0.0
        noise=max(r25,r12); d=abs(m25-m12)
        sn = ("%.1fx"%(d/noise)) if noise>1e-9 else "inf"
        print("%-18s %10.4f+-%.4f %10.4f+-%.4f | %9.4f %7.0fx %9.4f %s"
              %(arm,m25,r25,m12,r12,d,d/C1,noise,sn))

print("\n"+"="*100)
print("LIKE-FOR-LIKE: same 25.0 -> 12.5 step, same arms, both cards  (does the card move the conclusion?)")
print("="*100)
print("%-16s %6s | %9s %9s %9s %7s" % ("arm","hit","old d","new d","new rep","old xC1"))
for arm in ["g1_grid_prod","p5_penalty"]:
    for hit in [1,5,10,17]:
        o=[next((r["iou"] for r in old if r["hit"]==hit and r["arm"]==arm and r["speed"]==s),None) for s in (25.0,12.5)]
        nv={}
        for s in (25.0,12.5):
            vals=[r["iou"] for r in new if r["hit"]==hit and r["speed"]==s and base(r["arm"])==arm and r["arm"].endswith(("_r1","_r2"))]
            if vals: nv[s]=st.mean(vals)
        if any(x is None for x in o) or len(nv)<2: continue
        od=abs(o[0]-o[1]); nd=abs(nv[25.0]-nv[12.5])
        reps=[r["iou"] for r in new if r["hit"]==hit and base(r["arm"])==arm and r["arm"].endswith(("_r1","_r2"))]
        rep=max(reps)-min(reps)
        print("%-16s %6d | %9.4f %9.4f %9.4f %7.0fx" % (arm,hit,od,nd,rep,od/C1))

print("\n"+"="*100)
print("REPLICATE NOISE FLOOR measured here (new card, r1 vs r2, all arms/speeds/hits)")
print("="*100)
g=defaultdict(list)
for r in new:
    if r["arm"].endswith(("_r1","_r2")): g[(r["hit"],r["speed"],base(r["arm"]))].append(r["iou"])
sp=[max(v)-min(v) for v in g.values() if len(v)==2]
print("n pairs=%d  median |r1-r2| = %.5f   mean = %.5f   max = %.5f"%(len(sp),st.median(sp),st.mean(sp),max(sp)))
print("A5 published sigma = 0.00024 (res 7, hit 1, single config)")
o6=[r["iou"] for r in R if r["set"] in("old","old_rep") and r["speed"]==6.25 and r["arm"]=="g1_grid_prod" and r["hit"]==17]
if len(o6)==2: print("old-card independent repeat @6.25 hit17 (batch_speed_6p25 vs batch_speed2_6p25): %.4f vs %.4f  -> %.5f"%(o6[0],o6[1],abs(o6[0]-o6[1])))
