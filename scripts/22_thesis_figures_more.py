# -*- coding: utf-8 -*-
# 논문 본문 그림 중 21번에 없던 그림 9~19 (SHAP 상위 특징, 경보 예시, 서론·데이터·방법 장 그림)
# 입력: data/results/extra_fac_*, data/cleaned, data/target, data/eda/device_summary.csv
# 출력: charts/thesis_fig9~19 .png(300dpi), .pdf
#
# 사용법: python scripts/22_thesis_figures_more.py
import pandas as pd, numpy as np, glob, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(PROJECT_ROOT, "data") + os.sep; R = D + "results" + os.sep; OUT = os.path.join(PROJECT_ROOT, "charts") + os.sep
BLUE,ORANGE,INK,MUTED,GRID,RULE,RED="#2a78d6","#eb6834","#0b0b0b","#52514e","#e4e3df","#9a9893","#c8312b"
LBLUE="#bcd6f3"
def pick_font(names):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), "DejaVu Sans")
plt.rcParams["font.family"]=pick_font(["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"])
plt.rcParams.update({"axes.unicode_minus":False,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,
 "ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,
 "grid.linewidth":0.6,"axes.axisbelow":True,"font.size":9,"legend.frameon":False,"pdf.fonttype":42})
W=6.3
def save(fig,n):
    for e in ("png","pdf"): fig.savefig(OUT+n+"."+e,dpi=300,bbox_inches="tight")
    plt.close(fig)
PUMP='1.펌프_일반모터'
def load_val(dev):
    c=pd.read_parquet(f'{D}cleaned/Validation/{PUMP}/{dev}.parquet')
    t=pd.read_parquet(f'{D}target/Validation/{PUMP}/{dev}.parquet')
    return c.join(t,how='inner')

# 그림 5-11: 최종 모델 SHAP 상위 10개
s=pd.read_csv(R+'extra_fac_main_shap.csv').head(10)
names={"마지막이상후_분":"마지막 이상 후 경과 시간","이상비율_1440":"이상 비율 (24시간)","전류고조파최대상_max60":"최대 상 전류고조파 최댓값 (60분)",
"전류고조파평균_max10":"전류고조파 최댓값 (10분)","전류고조파최대상":"최대 상 전류고조파 (현재)","전류고조파최대상_max30":"최대 상 전류고조파 최댓값 (30분)",
"이상이력없음":"이상 이력 없음","전류고조파_q90_1440":"전류고조파 90% 분위수 (24시간)","전류평균_max60":"전류 최댓값 (60분)","시":"시각"}
extra={"이상비율_1440","전류고조파최대상_max60","전류고조파최대상","전류고조파최대상_max30","전류고조파_q90_1440"}
s=s.iloc[::-1]
fig,ax=plt.subplots(figsize=(W,3.6))
c=[BLUE if f in extra else RULE for f in s.feature]
ax.barh([names[f] for f in s.feature],s.share*100,color=c,height=0.7)
for y,v in enumerate(s.share*100): ax.text(v+0.3,y,f"{v:.1f}",va="center",fontsize=8,color=INK)
ax.set_xlabel("예측 기여 비중 (%, 평균 |SHAP|)"); ax.grid(axis="y",visible=False)
ax.legend(handles=[Patch(color=RULE,label="기본 특징"),Patch(color=BLUE,label="보강 특징")],loc="lower right")
save(fig,"thesis_fig9_shap_final")
# 그림 5-14: 경보 예시 (설비 241, 2020-12-17 이상 발생 24분 전 경보)
sc=pd.read_parquet(R+'extra_fac_v2cmp_scores.parquet')
th=pd.read_csv(R+'extra_fac_v2cmp_thresholds.csv',encoding='utf-8-sig'); th=th[th.column=='score_2'].set_index('fold').threshold
ev=pd.Timestamp('2020-12-17 21:42'); t0,t1=ev-pd.Timedelta('70min'),ev+pd.Timedelta('15min')
d=sc[(sc.dev==241)&(sc.time>=t0)&(sc.time<ev)].sort_values('time'); thr=th[d.fold.iloc[0]]
cl=pd.read_parquet(f'{D}cleaned/Validation/{PUMP}/241.parquet')
cl=cl.loc[t0:t1]
al=ev-pd.Timedelta('24min')
fig,(a1,a2)=plt.subplots(2,1,figsize=(W,3.8),sharex=True,gridspec_kw={"height_ratios":[1,1.2],"hspace":0.12})
a1.plot(cl.index,cl['전류고조파평균'],color=INK,lw=1.1); a1.axhline(30,color=RED,ls="--",lw=0.9)
a1.text(t0,31,"주의 경계 30",color=RED,fontsize=8,va="bottom"); a1.set_ylabel("전류고조파 (%)")
a2.plot(d.time,d.score_2,color=BLUE,lw=1.3,label="모델 점수"); a2.axhline(thr,color=MUTED,ls="--",lw=0.9,label=f"경보 기준 ({thr:.2f})")
a2.set_ylim(0,1); a2.set_ylabel("모델 점수")
for a in (a1,a2):
    a.axvline(al,color=ORANGE,lw=1.4); a.axvline(ev,color=RED,lw=1.4)
a2.text(al,0.05,"  경보 21:18",color=ORANGE,fontsize=8); a2.text(ev,0.05,"  이상 시작 21:42",color=RED,fontsize=8)
a2.legend(loc="upper left",bbox_to_anchor=(0,1.0)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
print("thr",thr, "rows",len(d), d.score_2.max())
save(fig,"thesis_fig10_alarm_example")

# 그림 1-1: 이상 발생 직전 예시
best=None
for f in sorted(glob.glob(f'{D}target/Validation/{PUMP}/*.parquet')):
    dev=os.path.basename(f)[:-8]; d=load_val(dev); h=d['전류고조파평균']
    for ev in d.index[d.event_start==1]:
        w=d.loc[ev-pd.Timedelta(minutes=30):ev-pd.Timedelta(minutes=1)]
        if len(w)<30 or w.eligible.min()<1 or w['전류고조파평균'].isna().any(): continue
        pre=d.loc[ev-pd.Timedelta(minutes=90):ev-pd.Timedelta(minutes=31)]
        if len(pre)<60 or pre.is_abnormal.max()>0 or pre.is_running.min()<1 or pre['전류고조파평균'].max()>27: continue
        if w['전류고조파평균'].diff().abs().max()>6: continue
        rise=w['전류고조파평균'].iloc[-5:].mean()-pre['전류고조파평균'].mean()
        if best is None or rise>best[0]: best=(rise,dev,ev)
rise,dev,ev=best; d=load_val(dev)
w=d.loc[ev-pd.Timedelta(minutes=90):ev+pd.Timedelta(minutes=20)]
x=(w.index-ev).total_seconds()/60
fig,ax=plt.subplots(figsize=(W,2.8))
ax.axvspan(-30,0,color=LBLUE,alpha=0.45,lw=0)
ax.text(-15,0.97,"이상 발생 전 30분\n(이 구간에서 미리 알리는 것이 목표)",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=8,color=INK)
ab=w.is_abnormal.values==1
ax.plot(x,w['전류고조파평균'].where(~ab),color=BLUE,lw=1.4,label="전류고조파평균 (라벨 정상)")
ax.plot(x,w['전류고조파평균'].where(ab),color=RED,lw=1.4,label="전류고조파평균 (라벨 주의 이상)")
ax.axhline(30,color=MUTED,ls="--",lw=0.9); ax.text(x.min()+1,30.6,"주의 경계 30",fontsize=8,color=MUTED)
ax.axvline(0,color=INK,lw=0.8); ax.text(0.8,0.04,"이상 발생 시작",transform=ax.get_xaxis_transform(),fontsize=8)
ax.set_xlabel("이상 발생 시작 기준 시간 (분)"); ax.set_ylabel("전류고조파평균 (%)")
ax.legend(loc="upper left",fontsize=8,bbox_to_anchor=(0,0.80))
save(fig,"thesis_fig11_intro_example"); print("1-1",dev,ev,round(rise,1))

# 그림 3-1: 설비별 수집 기간
ds=pd.read_csv(D+'eda/device_summary.csv',encoding='utf-8-sig',parse_dates=['start','end'])
order=ds.groupby(['facility','dev']).start.min().reset_index().sort_values(['facility','start','dev']).reset_index(drop=True)
pos={(r.facility,r.dev):i for i,r in order.iterrows()}
fig,ax=plt.subplots(figsize=(W,4.6))
for _,r in ds.iterrows():
    y=pos[(r.facility,r.dev)]
    ax.plot([r.start,r.end],[y,y],color=BLUE if r.split=='Training' else ORANGE,lw=1.6,solid_capstyle="butt")
nP=(order.facility==PUMP).sum()
ax.axhline(nP-0.5,color=MUTED,lw=0.7,ls=":")
ax.text(1.01,(nP-1)/2,f"펌프·일반모터\n{nP}대",transform=ax.get_yaxis_transform(),va="center",fontsize=8)
ax.text(1.01,nP+(len(order)-nP)/2,f"공조설비\n{len(order)-nP}대",transform=ax.get_yaxis_transform(),va="center",fontsize=8)
ax.set_yticks([]); ax.set_ylim(len(order),-1); ax.set_ylabel("설비 (설비종류·수집 시작일 순)")
ax.grid(axis="y",visible=False)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.legend(handles=[Patch(color=BLUE,label="Training"),Patch(color=ORANGE,label="Validation")],loc="lower left",ncol=2)
save(fig,"thesis_fig12_collection")

# 그림 3-3: 데이터 선별 과정
steps=[("원본 데이터","설비 105대 × Training·Validation\n라벨 종류별 JSON 5개, 세로형 (시각, 항목, 값)"),
       ("1차 전처리","시각 × 측정 항목 35개 가로형 표 + 라벨 5개\n210개 파일 변환"),
       ("2차 전처리","1분 격자, 측정 상한·전압 튐·주파수 이상 결측, 5분 이하 보간\n전원 꺼짐 95% 이상 파일 제외 → 209개 파일, 4,008,999행"),
       ("기준 라벨 선택","SOH 라벨 3종 중 전류고조파 라벨\n(역률: 정지 여부 반영, 전압고조파: 계통 영향)"),
       ("분석 대상","펌프·일반모터 73대\n(공조설비 32대는 양성 Training 19건, Validation 0건으로 제외)")]
fig,ax=plt.subplots(figsize=(W,4.4)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(-0.02,1.02)
n=len(steps); h=0.14; gap=(1-n*h)/(n-1)
for i,(t,b) in enumerate(steps):
    y=1-i*(h+gap)-h
    last=i==n-1
    ax.add_patch(FancyBboxPatch((0.02,y),0.96,h,boxstyle="round,pad=0.005,rounding_size=0.015",
        fc=LBLUE if last else "#f4f3ef",ec=BLUE if last else RULE,lw=0.9))
    ax.text(0.05,y+h/2,t,va="center",fontsize=9.5,fontweight="bold",color=INK)
    ax.text(0.27,y+h/2,b,va="center",fontsize=8.3,color=INK,linespacing=1.4)
    if i<n-1: ax.annotate("",xy=(0.5,y-gap+0.005),xytext=(0.5,y-0.005),arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=0.9))
save(fig,"thesis_fig13_pipeline")

# 그림 3-6: 양성 수 / 설비별 사건 수
ts=pd.read_csv(D+'target/target_summary.csv',encoding='utf-8-sig')
fig,(a1,a2)=plt.subplots(1,2,figsize=(W,2.9),gridspec_kw={"width_ratios":[1,2.2]})
g=ts[ts.facility.isin([PUMP,'2.공조설비'])].groupby(['facility','split']).pos_30.sum().unstack()
xs=np.arange(2); lab=["펌프·일반모터","공조설비"]
for k,(sp,col) in enumerate([("Training",BLUE),("Validation",ORANGE)]):
    v=g.loc[[PUMP,'2.공조설비'],sp].values
    a1.bar(xs+(k-0.5)*0.36,np.maximum(v,0.8),0.36,color=col,label=sp)
    for xx,vv in zip(xs+(k-0.5)*0.36,v): a1.text(xx,max(vv,0.8)*1.15,f"{vv:,}",ha="center",fontsize=7.5)
a1.set_yscale("log"); a1.set_ylim(0.7,2e5); a1.set_xticks(xs,lab); a1.set_ylabel("30분 내 이상 발생 양성 수 (로그)")
a1.legend(fontsize=7.5,loc="upper right"); a1.grid(axis="x",visible=False); a1.set_title("(a) 설비종류별 양성 수",fontsize=9,loc="left")
p=ts[(ts.split=='Training')&(ts.facility==PUMP)].sort_values('events',ascending=False)
p=p[p.events>0]
a2.bar(range(len(p)),p.events,color=[BLUE if i<5 else RULE for i in range(len(p))],width=0.8)
a2.set_xlabel(f"설비 (사건 수 순, 사건 있는 {len(p)}대)"); a2.set_ylabel("Training 이상 발생 사건 수"); a2.set_xticks([])
share=p.events.head(5).sum()/p.events.sum()
a2.text(5.5,p.events.iloc[0]*0.85,f"상위 5대가 전체 사건의 {share*100:.0f}%",fontsize=8,color=BLUE)
a2.grid(axis="x",visible=False); a2.set_title("(b) 펌프·일반모터 설비별 사건 수",fontsize=9,loc="left")
fig.tight_layout(); save(fig,"thesis_fig14_events")

# 그림 4-1: 예측 문제 정의
fig,ax=plt.subplots(figsize=(W,2.5)); ax.set_xlim(-66,36); ax.set_ylim(-0.3,4.4)
ax.grid(False); ax.spines["left"].set_visible(False); ax.set_yticks([])
for i,(wn,c) in enumerate([(60,"#dfe9f6"),(30,"#bcd6f3"),(10,"#8fbbea")]):
    ax.add_patch(Rectangle((-wn,2.6-i*0.62),wn,0.5,fc=c,ec="none"))
    if wn==10: ax.text(-wn-1,2.85-i*0.62,f"입력 창 {wn}분",va="center",ha="right",fontsize=7.8)
    else: ax.text(-wn+1,2.85-i*0.62,f"입력 창 {wn}분",va="center",fontsize=7.8)
ax.add_patch(Rectangle((0,2.6),30,0.5,fc="#fde3d6",ec="none"))
ax.text(15,2.85,"예측 구간 30분",ha="center",va="center",fontsize=8)
ax.add_patch(Rectangle((18,1.98),12,0.5,fc=RED,alpha=0.85,ec="none"))
ax.text(24,2.23,"이상 발생",ha="center",va="center",fontsize=7.8,color="white")
ax.plot([0,0],[-0.3,3.7],color=INK,lw=1); ax.text(0,3.95,"예측 시점 t\n(운전 중, 라벨 정상)",ha="center",fontsize=8)
ax.annotate("",xy=(18,1.5),xytext=(0,1.5),arrowprops=dict(arrowstyle="<->",color=MUTED,lw=0.8))
ax.text(9,1.1,"리드 타임",ha="center",fontsize=7.8,color=MUTED)
ax.text(-65,0.6,"각 창마다 평균·표준편차·최댓값·변화량을 계산한다.",fontsize=7.8,color=MUTED)
ax.text(31,0.6,"→ 시점 t의 정답 = 1 (양성)",fontsize=8,ha="right",color=INK)
ax.set_xlabel("예측 시점 기준 시간 (분)")
save(fig,"thesis_fig15_problem")

# 그림 4-2: 이상 구간 길이 분포
seg=[]
for f in glob.glob(f'{D}target/Training/{PUMP}/*.parquet'):
    t=pd.read_parquet(f); a=((t.is_abnormal==1)&(t.is_running==1)).astype(int).values
    dd=np.diff(np.r_[0,a,0]); seg+=list(np.where(dd==-1)[0]-np.where(dd==1)[0])
seg=np.array(seg)
fig,ax=plt.subplots(figsize=(W,2.6))
bins=np.unique(np.round(np.logspace(0,np.log10(seg.max()+1),40)))
ax.hist(seg,bins=bins,color=BLUE,ec="white",lw=0.4)
ax.set_xscale("log"); ax.axvline(5,color=RED,ls="--",lw=0.9)
ax.text(5.3,0.92,f"5분 기준\n5분 미만 {np.mean(seg<5)*100:.0f}%",transform=ax.get_xaxis_transform(),fontsize=8,color=RED,va="top")
ax.set_xlabel("운전 중 이상 구간 길이 (분, 로그)"); ax.set_ylabel("구간 수"); ax.grid(axis="x",visible=False)
save(fig,"thesis_fig16_seglen"); print("4-2",len(seg),np.mean(seg<5))

# 그림 4-3: 최근 60분 전류고조파 최댓값
vals={0:[],1:[]}
for f in sorted(glob.glob(f'{D}target/Validation/{PUMP}/*.parquet')):
    dev=os.path.basename(f)[:-8]; d=load_val(dev)
    m60=d['전류고조파평균'].rolling(60,min_periods=30).max()
    k=(d.eligible==1)&d.target_30.notna()&m60.notna()
    for c in (0,1): vals[c]+=list(m60[k&(d.target_30==c)].values)
fig,ax=plt.subplots(figsize=(W,2.6)); b=np.linspace(0,60,61)
ax.hist(np.clip(vals[0],0,60),bins=b,density=True,color=RULE,alpha=0.75,label=f"30분 내 이상 없음 (n={len(vals[0]):,})")
ax.hist(np.clip(vals[1],0,60),bins=b,density=True,color=BLUE,alpha=0.75,label=f"30분 내 이상 발생 (n={len(vals[1]):,})")
ax.axvline(30,color=MUTED,ls="--",lw=0.9); ax.text(30.5,0.9,"경계 30",transform=ax.get_xaxis_transform(),fontsize=8,color=MUTED)
ax.set_xlabel("최근 60분 전류고조파평균 최댓값 (%, 60 이상은 60)"); ax.set_ylabel("비율 밀도"); ax.legend(fontsize=8)
ax.grid(axis="x",visible=False)
save(fig,"thesis_fig17_max60"); print("4-3",len(vals[0]),len(vals[1]),np.median(vals[0]),np.median(vals[1]))

# 그림 4-4: 설비 단위 교차검증
fig,ax=plt.subplots(figsize=(W,3.0)); ax.axis("off"); ax.set_xlim(-1.6,6.2); ax.set_ylim(-1.5,5.6)
for j in range(5): ax.text(j+0.5,5.25,f"묶음 {j+1}",ha="center",fontsize=8.5)
for i in range(5):
    y=4-i; ax.text(-0.15,y+0.4,f"{i+1}회차",ha="right",va="center",fontsize=8.5)
    for j in range(5):
        if i==j:
            ax.add_patch(Rectangle((j+0.05,y+0.05),0.9,0.7,fc=ORANGE,ec="white"))
            ax.text(j+0.5,y+0.4,"평가\n(Validation)",ha="center",va="center",fontsize=7,color="white")
        else:
            ax.add_patch(Rectangle((j+0.05,y+0.05),0.72,0.7,fc=BLUE,ec="white"))
            ax.add_patch(Rectangle((j+0.77,y+0.05),0.18,0.7,fc=LBLUE,ec="white"))
ax.text(5.15,4.4,"묶음마다\n약 13대",fontsize=7.5,va="center",color=MUTED)
hs=[Patch(color=BLUE,label="학습 (Training 앞 80%)"),Patch(color=LBLUE,label="임계값·조기 종료 (Training 뒤 20%)"),Patch(color=ORANGE,label="평가 (처음 보는 설비의 Validation)")]
ax.legend(handles=hs,loc="lower center",ncol=3,fontsize=7.5,bbox_to_anchor=(0.5,-0.02))
save(fig,"thesis_fig18_cv")

# 그림 5-2: PR 곡선
from sklearn.metrics import precision_recall_curve, average_precision_score
s=pd.read_parquet(D+'results/extra_fac_v2cmp_scores.parquet'); s=s[s.target_30.notna()]
names=[("score_2","LightGBM (펌프+추가 6종)",BLUE,"-"),("score_4","Random Forest (펌프+추가 6종)",ORANGE,"-"),
       ("score_1","LightGBM (펌프만)",BLUE,":"),("score_3","Logistic Regression (펌프+추가 6종)",MUTED,"-"),("score_0","임계값 규칙",RULE,"--")]
fig,ax=plt.subplots(figsize=(W*0.72,3.6))
for c,l,col,ls in names:
    p_,r_,_=precision_recall_curve(s.target_30,s[c]); ax.plot(r_,p_,color=col,ls=ls,lw=1.3,label=f"{l}  {average_precision_score(s.target_30,s[c]):.3f}")
base=s.target_30.mean(); ax.axhline(base,color=GRID,lw=1); ax.text(0.99,base+0.015,f"무작위 {base:.3f}",ha="right",fontsize=7.5,color=MUTED)
ax.set_xlabel("재현율"); ax.set_ylabel("정밀도"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.legend(title="모델  PR-AUC",fontsize=7.3,title_fontsize=7.5,loc="upper right")
save(fig,"thesis_fig19_pr_curve")
