# -*- coding: utf-8 -*-
# 졸업논문 양식(2단, 단 너비 81mm)에 맞춘 그림 26개. 학교 양식이 그림 안 표기를 영문으로 요구해 22번과 따로 둠
# 입력: data/results, data/cleaned, data/target, data/processed, data/eda
# 출력: charts/en/fig01~fig26.png(400dpi)
#
# 사용법: python scripts/23_thesis_figures_en.py
import pandas as pd, numpy as np, glob, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib import font_manager
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(PROJECT_ROOT, "data") + os.sep; R = D + "results" + os.sep; OUT = os.path.join(PROJECT_ROOT, "charts", "en") + os.sep
os.makedirs(OUT, exist_ok=True)
def pick_font(names):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), "DejaVu Sans")
BLUE,ORANGE,AQUA,VIOLET="#2a78d6","#eb6834","#1baf7a","#6250d6"
INK,MUTED,GRID,RULE,RED,LBLUE="#0b0b0b","#52514e","#e4e3df","#9a9893","#c8312b","#bcd6f3"
GOOD,WARN,CRIT="#2a78d6","#e8a33d","#c8312b"
plt.rcParams.update({"font.family":pick_font(["Liberation Sans", "Arial", "Helvetica"]),"axes.unicode_minus":False,"axes.edgecolor":MUTED,"axes.labelcolor":INK,
 "xtick.color":MUTED,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,
 "grid.linewidth":0.5,"axes.axisbelow":True,"font.size":6.8,"axes.labelsize":6.8,"xtick.labelsize":6.3,"ytick.labelsize":6.3,
 "legend.fontsize":6,"legend.frameon":False,"lines.linewidth":1.1,"pdf.fonttype":42,"axes.linewidth":0.6,
 "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.5,"ytick.major.size":2.5})
W=3.15
def save(fig,n):
    fig.savefig(OUT+n+".png",dpi=400,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
def rd(n): return pd.read_csv(R+n,encoding='utf-8-sig')
PUMP='1.펌프_일반모터'
PO,PE="LightGBM 펌프만","LightGBM 펌프+추가"
MN={"규칙":"Threshold rule","Logistic Regression 펌프+추가":"LR (pump+6 types)",PO:"LightGBM (pump only)",
    "Random Forest 펌프+추가":"RF (pump+6 types)",PE:"LightGBM (pump+6 types)"}
MODELS=[("규칙",RULE),("Logistic Regression 펌프+추가",AQUA),(PO,ORANGE),("Random Forest 펌프+추가",VIOLET),(PE,BLUE)]
TYPES={"3":"Air compressor","4":"Machine tool","6":"Heating furnace","7":"Extruder","9":"Casting machine","10":"Heat treatment"}
FAC={"1":"Pump/motor","2":"HVAC","3":"Air compressor","4":"Machine tool","5":"Boiler","6":"Heating furnace","7":"Extruder","8":"Injection molding","9":"Casting machine","10":"Heat treatment"}
def load_val(dev):
    c=pd.read_parquet(f'{D}cleaned/Validation/{PUMP}/{dev}.parquet'); t=pd.read_parquet(f'{D}target/Validation/{PUMP}/{dev}.parquet')
    return c.join(t,how='inner')
H='전류고조파평균'

# 그림 1 (1-1): 이상 발생 직전 예시
d=load_val('300'); ev=pd.Timestamp('2021-02-04 09:11:00')
w=d.loc[ev-pd.Timedelta(minutes=90):ev+pd.Timedelta(minutes=20)]; x=(w.index-ev).total_seconds()/60; ab=w.is_abnormal.values==1
fig,ax=plt.subplots(figsize=(W,1.9))
ax.axvspan(-30,0,color=LBLUE,alpha=0.45,lw=0)
ax.text(-15,0.97,"30 min before onset\n(target warning window)",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=6)
ax.plot(x,w[H].where(~ab),color=BLUE,label="Label: normal"); ax.plot(x,w[H].where(ab),color=RED,label="Label: caution")
ax.axhline(30,color=MUTED,ls="--",lw=0.7); ax.text(-89,30.8,"Caution boundary (30)",fontsize=6,color=MUTED)
ax.axvline(0,color=INK,lw=0.7); ax.text(1,0.04,"Onset",transform=ax.get_xaxis_transform(),fontsize=6)
ax.set_xlabel("Time from anomaly onset (min)"); ax.set_ylabel("Mean current THD (%)")
ax.legend(loc="upper left",bbox_to_anchor=(0,0.78)); save(fig,"fig01")

# 그림 2 (3-1): 설비별 수집 기간
ds=pd.read_csv(D+'eda/device_summary.csv',encoding='utf-8-sig',parse_dates=['start','end'])
order=ds.groupby(['facility','dev']).start.min().reset_index().sort_values(['facility','start','dev']).reset_index(drop=True)
pos={(r.facility,r.dev):i for i,r in order.iterrows()}
fig,ax=plt.subplots(figsize=(W,2.9))
for _,r in ds.iterrows():
    ax.plot([r.start,r.end],[pos[(r.facility,r.dev)]]*2,color=BLUE if r.split=='Training' else ORANGE,lw=1.0,solid_capstyle="butt")
nP=(order.facility==PUMP).sum(); ax.axhline(nP-0.5,color=MUTED,lw=0.5,ls=":")
ax.text(1.01,(nP-1)/2,f"Pump/motor\n({nP} units)",transform=ax.get_yaxis_transform(),va="center",fontsize=6)
ax.text(1.01,nP+(len(order)-nP)/2,f"HVAC\n({len(order)-nP} units)",transform=ax.get_yaxis_transform(),va="center",fontsize=6)
ax.set_yticks([]); ax.set_ylim(len(order),-1); ax.set_ylabel("Unit (by type and start date)"); ax.grid(axis="y",visible=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.tick_params(axis='x',labelsize=5.8)
ax.legend(handles=[Patch(color=BLUE,label="Training"),Patch(color=ORANGE,label="Validation")],loc="lower left",ncol=2); save(fig,"fig02")

# 그림 3 (3-2): 센서 상한에서 잘린 기동 전류
p=pd.read_parquet(D+'processed/Training/1.펌프_일반모터/484.parquet')
if 'TIMESTAMP' in p.columns: p=p.set_index('TIMESTAMP')
p=p.sort_index().loc["2021-02-04 18:30":"2021-02-04 19:30"]
PH=['R상전류','S상전류','T상전류']
fig,ax=plt.subplots(figsize=(W,1.7))
ax.plot(p.index,p['전류평균'],color=BLUE)
hit=p[(p[PH]==512).any(axis=1)]
ax.scatter(hit.index,hit['전류평균'],s=14,color=CRIT,zorder=3,edgecolor="white",linewidth=0.8)
for t,v in hit['전류평균'].items(): ax.annotate("512 A (sensor limit)",(t,v),xytext=(5,-3),textcoords="offset points",fontsize=6)
ax.set_ylabel("Mean current (A)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); save(fig,"fig03")
print("clip hits",len(hit))

# 그림 4 (3-3): 데이터 선별 과정
steps=[("Raw data","105 units x Training/Validation\n5 JSON files per unit, long format"),
 ("Step 1","Wide table: time x 35 measurements + 5 labels\n210 files converted"),
 ("Step 2","1-min grid, invalid values to missing, short gaps\ninterpolated -> 209 files, 4,008,999 rows"),
 ("Label choice","Current-THD SOH label as the main label\n(PF: stop state; voltage THD: grid effect)"),
 ("Target units","73 pump/motor units (32 HVAC units\nexcluded: too few positives)")]
fig,ax=plt.subplots(figsize=(W,3.0)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(-0.02,1.02)
n=len(steps); h=0.15; gap=(1-n*h)/(n-1)
for i,(t,b) in enumerate(steps):
    y=1-i*(h+gap)-h; last=i==n-1
    ax.add_patch(FancyBboxPatch((0.01,y),0.98,h,boxstyle="round,pad=0.004,rounding_size=0.012",fc=LBLUE if last else "#f4f3ef",ec=BLUE if last else RULE,lw=0.6))
    ax.text(0.03,y+h/2,t,va="center",fontsize=5.8,fontweight="bold"); ax.text(0.33,y+h/2,b,va="center",fontsize=5.3,linespacing=1.3)
    if i<n-1: ax.annotate("",xy=(0.5,y-gap+0.004),xytext=(0.5,y-0.004),arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=0.6,mutation_scale=6))
save(fig,"fig04")

# 그림 5 (3-4): SOH 라벨별 전류고조파 분포, 구간별 라벨 비율 (Validation 펌프, 운전 중)
rows=[]
for f in sorted(glob.glob(f'{D}cleaned/Validation/{PUMP}/*.parquet')):
    c=pd.read_parquet(f); c=c[c['label_기동_전류평균']!='Stop']; rows.append(c[[H,'label_SOH_전류고조파평균']])
v=pd.concat(rows).dropna().reset_index(drop=True); lab=v['label_SOH_전류고조파평균']
fig,(a1,a2)=plt.subplots(2,1,figsize=(W,3.1),gridspec_kw={"hspace":0.55})
b=np.linspace(0,80,81)
for L,cn,col in [("정상","Normal",GOOD),("주의","Caution",WARN),("경고","Warning",CRIT)]:
    a1.hist(v.loc[lab==L,H].clip(0,80),bins=b,color=col,alpha=0.75,label=cn)
for xx in (30,60): a1.axvline(xx,color=INK,ls="--",lw=0.6)
a1.set_yscale("log"); a1.set_xlabel("Mean current THD (%)"); a1.set_ylabel("Minutes (log)"); a1.legend(ncol=3,loc="upper right"); a1.set_title("(a) Distribution by SOH label",loc="left",fontsize=6.8)
bins=[0,10,20,25,28,30,32,35,40,60,1e9]; names=["0-10","10-20","20-25","25-28","28-30","30-32","32-35","35-40","40-60","60+"]
cut=pd.cut(v[H],bins,right=False,labels=names); sh=pd.crosstab(cut,lab,normalize='index').reindex(columns=["정상","주의","경고"]).fillna(0)
left=np.zeros(len(sh))
for L,cn,col in [("정상","Normal",GOOD),("주의","Caution",WARN),("경고","Warning",CRIT)]:
    a2.bar(range(len(sh)),sh[L],bottom=left,color=col,width=0.75,label=cn); left+=sh[L].values
a2.set_xticks(range(len(sh)),names,rotation=35,fontsize=5.6); a2.set_ylim(0,1); a2.yaxis.set_major_formatter(lambda q,_:f"{q:.0%}")
a2.set_xlabel("Mean current THD bin (%)"); a2.set_ylabel("Label share"); a2.grid(axis="x",visible=False); a2.set_title("(b) Label share by THD bin",loc="left",fontsize=6.8)
save(fig,"fig05"); print("fig5 share",sh.loc[["28-30","30-32"]].round(3).to_dict())

# 그림 6 (3-5): 기동 상태별 SOH 라벨 비율
s=pd.read_csv(D+'eda/soh_by_state.csv',header=[0,1],index_col=0,encoding='utf-8-sig')
s=s.drop(index=[i for i in s.index if str(i).startswith('label_')],errors='ignore')
fig,axes=plt.subplots(3,1,figsize=(W,2.7),sharex=True,gridspec_kw={"hspace":0.55})
for ax,(col,tt) in zip(axes,[("label_SOH_역률평균","Power factor label"),("label_SOH_전류고조파평균","Current THD label"),("label_SOH_전압고조파평균","Voltage THD label")]):
    t=s[col].reindex(["Stop","Unloading","Loading"]).astype(float); left=np.zeros(3)
    for L,cn,c in [("정상","Normal",GOOD),("주의","Caution",WARN),("경고","Warning",CRIT)]:
        ax.barh(range(3),t[L],left=left,color=c,height=0.65,label=cn,edgecolor="white",linewidth=0.5)
        for yy,(l,ww) in enumerate(zip(left,t[L])):
            if ww>0.1: ax.text(l+ww/2,yy,f"{ww:.0%}",ha="center",va="center",fontsize=5.5,color="white")
        left+=t[L].values
    ax.set_yticks(range(3),["Stop","Unloading","Loading"]); ax.set_title(tt,loc="left",fontsize=6.5); ax.grid(axis="y",visible=False)
axes[-1].legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.35)); axes[-1].set_xlim(0,1); axes[-1].xaxis.set_major_formatter(lambda q,_:f"{q:.0%}")
save(fig,"fig06")

# 그림 7 (3-6): 설비종류별 양성 수, 설비별 사건 수
ts=pd.read_csv(D+'target/target_summary.csv',encoding='utf-8-sig')
fig,(a1,a2)=plt.subplots(1,2,figsize=(W,1.9),gridspec_kw={"width_ratios":[1,1.6],"wspace":0.45})
g=ts[ts.facility.isin([PUMP,'2.공조설비'])].groupby(['facility','split']).pos_30.sum().unstack()
for k,(sp,col) in enumerate([("Training",BLUE),("Validation",ORANGE)]):
    vv=g.loc[[PUMP,'2.공조설비'],sp].values; xs=np.arange(2)+(k-0.5)*0.38
    a1.bar(xs,np.maximum(vv,0.8),0.38,color=col,label=sp)
    for xx,q in zip(xs,vv): a1.text(xx,max(q,0.8)*1.2,f"{q:,}",ha="center",fontsize=5.2)
a1.set_yscale("log"); a1.set_ylim(0.7,3e8); a1.set_xticks([0,1],["Pump/\nmotor","HVAC"]); a1.set_ylabel("Positive time points (log)")
a1.legend(loc="upper right",fontsize=5.2); a1.grid(axis="x",visible=False); a1.set_title("(a)",loc="left",fontsize=6.8)
pp=ts[(ts.split=='Training')&(ts.facility==PUMP)].sort_values('events',ascending=False); pp=pp[pp.events>0]
a2.bar(range(len(pp)),pp.events,color=[BLUE if i<5 else RULE for i in range(len(pp))],width=0.8)
a2.set_xlabel(f"Pump/motor unit ({len(pp)} units with events)"); a2.set_ylabel("Anomaly onsets (Training)"); a2.set_xticks([])
a2.text(5.5,pp.events.iloc[0]*0.85,f"Top 5 units: {pp.events.head(5).sum()/pp.events.sum():.0%}",fontsize=6,color=BLUE)
a2.grid(axis="x",visible=False); a2.set_title("(b)",loc="left",fontsize=6.8); save(fig,"fig07")

# 그림 8 (4-1): 예측 문제 정의
fig,ax=plt.subplots(figsize=(W,1.75)); ax.set_xlim(-66,36); ax.set_ylim(-0.1,4.4)
ax.grid(False); ax.spines["left"].set_visible(False); ax.set_yticks([])
for i,(wn,c) in enumerate([(60,"#dfe9f6"),(30,"#bcd6f3"),(10,"#8fbbea")]):
    ax.add_patch(Rectangle((-wn,2.6-i*0.62),wn,0.5,fc=c,ec="none"))
    if wn==10: ax.text(-wn-1,2.85-i*0.62,f"{wn}-min window",va="center",ha="right",fontsize=5.8)
    else: ax.text(-wn+1,2.85-i*0.62,f"{wn}-min window",va="center",fontsize=5.8)
ax.add_patch(Rectangle((0,2.6),30,0.5,fc="#fde3d6",ec="none")); ax.text(15,2.85,"30-min horizon",ha="center",va="center",fontsize=5.8)
ax.add_patch(Rectangle((18,1.98),12,0.5,fc=RED,alpha=0.85,ec="none")); ax.text(24,2.23,"Onset",ha="center",va="center",fontsize=5.6,color="white")
ax.plot([0,0],[-0.1,3.7],color=INK,lw=0.8); ax.text(0,3.85,"Prediction time t\n(running, label normal)",ha="center",fontsize=5.8)
ax.annotate("",xy=(18,1.5),xytext=(0,1.5),arrowprops=dict(arrowstyle="<->",color=MUTED,lw=0.6,mutation_scale=6)); ax.text(9,1.05,"Lead time",ha="center",fontsize=5.8,color=MUTED)
ax.text(-65,0.45,"Mean, SD, max and change per window",fontsize=5.6,color=MUTED); ax.text(35,0.45,"-> label at t = 1",fontsize=5.8,ha="right")
ax.set_xlabel("Time from prediction point (min)"); save(fig,"fig08")

# 그림 9 (4-2): 이상 구간 길이
seg=[]
for f in glob.glob(f'{D}target/Training/{PUMP}/*.parquet'):
    t=pd.read_parquet(f); a=((t.is_abnormal==1)&(t.is_running==1)).astype(int).values; dd=np.diff(np.r_[0,a,0]); seg+=list(np.where(dd==-1)[0]-np.where(dd==1)[0])
seg=np.array(seg)
fig,ax=plt.subplots(figsize=(W,1.7)); bb=np.unique(np.round(np.logspace(0,np.log10(seg.max()+1),36)))
ax.hist(seg,bins=bb,color=BLUE,ec="white",lw=0.3); ax.set_xscale("log"); ax.axvline(5,color=RED,ls="--",lw=0.7)
ax.text(5.4,0.93,f"5 min\n{np.mean(seg<5):.0%} shorter",transform=ax.get_xaxis_transform(),fontsize=6,color=RED,va="top")
ax.set_xlabel("Abnormal segment length while running (min, log)"); ax.set_ylabel("Segments"); ax.grid(axis="x",visible=False); save(fig,"fig09")

# 그림 10 (4-3): 이상 여부별 최근 60분 최댓값
vals={0:[],1:[]}
for f in sorted(glob.glob(f'{D}target/Validation/{PUMP}/*.parquet')):
    dv=load_val(os.path.basename(f)[:-8]); m60=dv[H].rolling(60,min_periods=30).max(); k=(dv.eligible==1)&dv.target_30.notna()&m60.notna()
    for c in (0,1): vals[c]+=list(m60[k&(dv.target_30==c)].values)
fig,ax=plt.subplots(figsize=(W,1.8)); b=np.linspace(0,60,61)
ax.hist(np.clip(vals[0],0,60),bins=b,density=True,color=RULE,alpha=0.75,label=f"No onset in 30 min (n={len(vals[0]):,})")
ax.hist(np.clip(vals[1],0,60),bins=b,density=True,color=BLUE,alpha=0.75,label=f"Onset in 30 min (n={len(vals[1]):,})")
ax.axvline(30,color=MUTED,ls="--",lw=0.7); ax.set_xlabel("Max mean current THD in last 60 min (%, capped at 60)"); ax.set_ylabel("Density")
ax.legend(loc="upper center"); ax.grid(axis="x",visible=False); save(fig,"fig10")

# 그림 11 (4-4): 설비 단위 교차검증
fig,ax=plt.subplots(figsize=(W,2.0)); ax.axis("off"); ax.set_xlim(-1.3,5.1); ax.set_ylim(-1.6,5.6)
for j in range(5): ax.text(j+0.5,5.2,f"Group {j+1}",ha="center",fontsize=6)
for i in range(5):
    y=4-i; ax.text(-0.1,y+0.4,f"Fold {i+1}",ha="right",va="center",fontsize=6)
    for j in range(5):
        if i==j: ax.add_patch(Rectangle((j+0.05,y+0.05),0.9,0.7,fc=ORANGE,ec="white")); ax.text(j+0.5,y+0.4,"Test",ha="center",va="center",fontsize=5.6,color="white")
        else: ax.add_patch(Rectangle((j+0.05,y+0.05),0.72,0.7,fc=BLUE,ec="white")); ax.add_patch(Rectangle((j+0.77,y+0.05),0.18,0.7,fc=LBLUE,ec="white"))
hs=[Patch(color=BLUE,label="Train (first 80% of Training)"),Patch(color=LBLUE,label="Threshold/early stop (last 20%)"),Patch(color=ORANGE,label="Test (Validation of unseen units)")]
ax.legend(handles=hs,loc="lower center",ncol=1,fontsize=5.6,bbox_to_anchor=(0.5,-0.08)); save(fig,"fig11")

# 그림 12 (5-1): 모델별 PR-AUC (펌프만, 본 적 있는 설비)
mm=rd("model_metrics.csv").set_index("model"); base=rd("baseline_metrics.csv").set_index("model").loc["항상 음성","pr_auc"]
order=[("규칙: 현재 전류고조파","Threshold rule"),("Logistic Regression (이력 없음 중앙값)","LR (median fill)"),("Logistic Regression (이력 없음 표시)","LR (no-history flag)"),("Random Forest","Random Forest"),("LightGBM","LightGBM")]
fig,ax=plt.subplots(figsize=(W,1.7)); vv=[mm.loc[k,"pr_auc"] for k,_ in order]
ax.barh(range(5),vv,color=BLUE,height=0.55)
for i,q in enumerate(vv): ax.text(q+0.01,i,f"{q:.3f}",va="center",fontsize=6)
ax.axvline(base,color=MUTED,ls="--",lw=0.7); ax.text(base+0.01,-0.75,f"Random {base:.3f}",fontsize=5.6,color=MUTED)
ax.set_yticks(range(5),[n for _,n in order]); ax.set_ylim(4.5,-1.0); ax.set_xlim(0,0.85); ax.grid(axis="y",visible=False); ax.set_xlabel("PR-AUC (Validation, seen units)"); save(fig,"fig12")

# 그림 13 (5-2): 최종 조건 PR 곡선
from sklearn.metrics import precision_recall_curve, average_precision_score
sc=pd.read_parquet(R+'extra_fac_v2cmp_scores.parquet'); s2=sc[sc.target_30.notna()]
fig,ax=plt.subplots(figsize=(W,2.5))
for c,l,col,ls in [("score_2","LightGBM (pump+6 types)",BLUE,"-"),("score_4","RF (pump+6 types)",ORANGE,"-"),("score_1","LightGBM (pump only)",BLUE,":"),("score_3","LR (pump+6 types)",MUTED,"-"),("score_0","Threshold rule",RULE,"--")]:
    pr,rc,_=precision_recall_curve(s2.target_30,s2[c]); ax.plot(rc,pr,color=col,ls=ls,lw=1.0,label=f"{l}  {average_precision_score(s2.target_30,s2[c]):.3f}")
bs=s2.target_30.mean(); ax.axhline(bs,color=GRID,lw=0.8); ax.text(0.99,bs+0.015,f"Random {bs:.3f}",ha="right",fontsize=5.6,color=MUTED)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(title="Model  PR-AUC",title_fontsize=6,loc="upper right",fontsize=5.6); save(fig,"fig13")

# 그림 14 (5-3): 방식별 본 적 있는/처음 보는 설비
us=rd("unit_split_metrics.csv"); vf=rd("value_forecast_metrics.csv")
seen={"규칙":us[(us.setting=="본 적 있는 설비")&(us.model=="규칙")].pr_auc.iloc[0]}
def g2(df,setting,m): return df[(df.setting==setting)&(df.model==m)].pr_auc.iloc[0]
rows=[("Threshold rule",g2(us,"본 적 있는 설비","규칙"),g2(us,"처음 보는 설비","규칙")),
 ("Value forecast\n(prior approach)",g2(vf,"본 적 있는 설비","값 예측 후 판정 (기준값 30)"),g2(vf,"처음 보는 설비","값 예측 후 판정 (기준값 30)")),
 ("Logistic Regression",g2(us,"본 적 있는 설비","Logistic Regression"),g2(us,"처음 보는 설비","Logistic Regression")),
 ("Transformer",0.657,0.416),("LightGBM",g2(us,"본 적 있는 설비","LightGBM"),g2(us,"처음 보는 설비","LightGBM"))]
fig,ax=plt.subplots(figsize=(W,2.3)); hh=0.36
for j,(lab_,col) in enumerate([("Seen units",BLUE),("Unseen units",ORANGE)]):
    ys=[i+(j-0.5)*(hh+0.02) for i in range(len(rows))]; q=[r[1+j] for r in rows]
    ax.barh(ys,q,height=hh,color=col,label=lab_)
    for yy,qq in zip(ys,q): ax.text(qq+0.008,yy,f"{qq:.3f}",va="center",fontsize=5.6)
ax.axvline(base,color=INK,ls="--",lw=0.6,label=f"Random {base:.3f}")
ax.set_yticks(range(len(rows)),[r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0,0.85); ax.grid(axis="y",visible=False)
ax.set_xlabel("PR-AUC (onset within 30 min)"); ax.legend(loc="upper center",bbox_to_anchor=(0.4,-0.2),ncol=3,fontsize=5.6); save(fig,"fig14")

# 그림 15 (5-4): 본 적 있는/처음 보는 설비 (규칙, LR, LightGBM)
fig,ax=plt.subplots(figsize=(W,1.8)); ww=0.36; ms=["규칙","Logistic Regression","LightGBM"]; mlab=["Threshold rule","Logistic Regression","LightGBM"]
for j,(st,lab_,col) in enumerate([("본 적 있는 설비","Seen units",BLUE),("처음 보는 설비","Unseen units",ORANGE)]):
    xs=[i+(j-0.5)*(ww+0.02) for i in range(3)]; q=[g2(us,st,m) for m in ms]
    ax.bar(xs,q,width=ww,color=col,label=lab_)
    for xx,qq in zip(xs,q): ax.text(xx,qq+0.012,f"{qq:.3f}",ha="center",fontsize=5.8)
ax.set_xticks(range(3),mlab); ax.set_ylim(0,0.85); ax.grid(axis="x",visible=False); ax.set_ylabel("PR-AUC"); ax.legend(loc="upper left"); save(fig,"fig15")

def ci_row(df,model,group="전체"):
    r=df[(df["group"]==group)&(df["model"]==model)].iloc[0]; return r["pr_auc"],r["ci_low"],r["ci_high"]
def diff_row(df,a,b,group="전체"):
    r=df[(df["group"]==group)&(df["model_a"]==a)&(df["model_b"]==b)].iloc[0]; return r["diff"],r["ci_low"],r["ci_high"],r["p_b_better"]

# 그림 16 (5-5): 최종 조건 모델 비교와 신뢰구간
ci=rd("boot_v2cmp_ci.csv").sort_values("pr_auc")
fig,ax=plt.subplots(figsize=(W,1.7)); y=np.arange(len(ci)); cols=[dict(MODELS)[m] for m in ci.model]
ax.hlines(y,ci.ci_low,ci.ci_high,color=cols,lw=1.6); ax.scatter(ci.pr_auc,y,s=18,color=cols,zorder=3,edgecolor="white",linewidth=0.8)
for yi,(q,hi) in enumerate(zip(ci.pr_auc,ci.ci_high)): ax.text(hi+0.01,yi,f"{q:.3f}",va="center",fontsize=6)
ax.set_yticks(y,[MN[m] for m in ci.model]); ax.set_xlim(0,0.92); ax.grid(axis="y",visible=False); ax.set_xlabel("PR-AUC with 95% CI (60 unseen pump units)"); save(fig,"fig16")

# 그림 17 (5-6): 묶음별 결과
fo=rd("extra_fac_v2cmp_folds.csv")
fig,ax=plt.subplots(figsize=(W,1.8))
for m,c in [("규칙",RULE),(PO,ORANGE),(PE,BLUE)]:
    dd=fo[fo.model==m].sort_values("fold"); ax.plot(dd.fold,dd.pr_auc,"-o",color=c,ms=3.5,mec="white",mew=0.6,label=MN[m])
ax.set_xticks(range(5),[f"Group {i+1}" for i in range(5)]); ax.set_ylim(0,1); ax.set_ylabel("PR-AUC"); ax.grid(axis="x",visible=False)
ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.13),fontsize=5.6); save(fig,"fig17")

# 그림 18 (5-7): 예측 구간별 결과
tags=[("10 min","boot_v2cmp_h10_ci.csv"),("30 min","boot_v2cmp_ci.csv"),("60 min","boot_v2cmp_h60_ci.csv")]
fig,ax=plt.subplots(figsize=(W,2.1))
for k,(m,c) in enumerate(MODELS):
    xs=np.arange(3)+(k-2)*0.15; vv=[ci_row(rd(f),m) for _,f in tags]; q,lo,hi=map(np.array,zip(*vv))
    ax.errorbar(xs,q,yerr=[q-lo,hi-q],fmt="o",ms=3,color=c,ecolor=c,elinewidth=1.2,capsize=0,mec="white",mew=0.6,label=MN[m])
ax.set_xticks(range(3),[f"{t} horizon" for t,_ in tags]); ax.set_ylim(0,0.9); ax.set_ylabel("PR-AUC (95% CI)"); ax.grid(axis="x",visible=False)
ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.12),fontsize=5.6); save(fig,"fig18")

# 그림 19 (5-8): 설비종류 추가 이득
rows=[("H10, onset >= 5 min","boot_v2cmp_h10_diff.csv"),("H30, onset >= 5 min (main)","boot_v2cmp_diff.csv"),("H60, onset >= 5 min","boot_v2cmp_h60_diff.csv"),("H30, onset >= 3 min","boot_v2_d3_diff.csv"),("H30, onset >= 10 min","boot_v2_d10_diff.csv")]
fig,ax=plt.subplots(figsize=(W,1.8))
for i,(lab_,f) in enumerate(rows[::-1]):
    dd,lo,hi,pp_=diff_row(rd(f),PO,PE); c=BLUE if lo>0 else MUTED
    ax.hlines(i,lo,hi,color=c,lw=1.6); ax.scatter(dd,i,s=16,color=c,zorder=3,edgecolor="white",linewidth=0.6)
    ax.text(0.335,i,f"+{dd:.3f} [{lo:.3f}, {hi:.3f}]",va="center",fontsize=5.6)
ax.axvline(0,color=MUTED,lw=0.6); ax.set_yticks(range(5),[r[0] for r in rows[::-1]]); ax.set_xlim(-0.1,0.33); ax.grid(axis="y",visible=False)
ax.set_xlabel("PR-AUC gain (pump only -> pump+6 types)"); save(fig,"fig19")

# 그림 20 (5-9): 설비종류 하나씩 빼기
m0=rd("extra_fac_v2cmp_metrics.csv").set_index("model")["pr_auc"]; bse,only=m0[PE],m0[PO]
rr=sorted([(f"Without {TYPES[k].lower()}",rd(f"extra_fac_v2_lo_{k}_metrics.csv").set_index("model")["pr_auc"][PE]) for k in TYPES],key=lambda r:r[1])
fig,ax=plt.subplots(figsize=(W,1.9)); y=np.arange(len(rr)); q=np.array([r[1] for r in rr])
ax.barh(y,q,height=0.6,color=BLUE)
for yi,qq in zip(y,q): ax.text(qq-0.006,yi,f"{qq:.3f} ({qq-bse:+.3f})",va="center",ha="right",fontsize=5.6,color="white")
ax.axvline(bse,color=INK,lw=0.7,ls="--"); ax.axvline(only,color=ORANGE,lw=0.7,ls="--")
ax.text(bse,len(rr)-0.4,f" All 6 types {bse:.3f}",fontsize=5.6,va="bottom"); ax.text(only,len(rr)-0.4,f"Pump only {only:.3f} ",fontsize=5.6,color=ORANGE,va="bottom",ha="right")
ax.set_yticks(y,[r[0] for r in rr]); ax.set_xlim(0,0.75); ax.set_ylim(-0.6,len(rr)+0.3); ax.grid(axis="y",visible=False); ax.set_xlabel("PR-AUC (LightGBM, pump+other types)"); save(fig,"fig20")

# 그림 21 (5-10): 설비종류별 결과
ci5,df5=rd("boot_v2s_ci.csv"),rd("boot_v2s_diff.csv"); same,pooled="LightGBM 같은 종류만","LightGBM 전체 종류"
groups=sorted(ci5.group.unique(),key=lambda g:int(g.split(".")[0]))[::-1]
fig,(a1,a2)=plt.subplots(1,2,figsize=(W,2.4),sharey=True,gridspec_kw={"width_ratios":[1.25,1],"wspace":0.08})
y=np.arange(len(groups)); units=ci5.drop_duplicates("group").set_index("group")["units"]
for m,c,off,l in [("규칙",RULE,0.22,"Rule"),(same,ORANGE,0,"Same type only"),(pooled,BLUE,-0.22,"All types")]:
    vv=np.array([ci_row(ci5,m,g_) for g_ in groups])
    a1.errorbar(vv[:,0],y+off,xerr=[vv[:,0]-vv[:,1],vv[:,2]-vv[:,0]],fmt="o",ms=2.6,color=c,ecolor=c,elinewidth=1,capsize=0,label=l)
a1.set_yticks(y,[f"{FAC[g_.split('.')[0]]} ({units[g_]})" for g_ in groups]); a1.set_xlim(0,1); a1.set_xlabel("PR-AUC (95% CI)"); a1.grid(axis="y",visible=False)
a1.legend(ncol=3,loc="lower left",bbox_to_anchor=(0,1.0),fontsize=5.4,handletextpad=0.2,columnspacing=0.6)
for i,g_ in enumerate(groups):
    dd,lo,hi,pp_=diff_row(df5,same,pooled,g_); c=BLUE if lo>0 else MUTED
    a2.hlines(i,lo,hi,color=c,lw=1.4); a2.scatter(dd,i,s=12,color=c,zorder=3)
a2.axvline(0,color=MUTED,lw=0.6); a2.set_xlim(-0.1,0.33); a2.set_xlabel("Gain (same type -> all types)"); a2.grid(axis="y",visible=False)
save(fig,"fig21")

# 그림 22 (5-11): 최종 모델 SHAP 상위 10개
sv=rd('extra_fac_main_shap.csv').head(10)
nm={"마지막이상후_분":"Minutes since last anomaly","이상비율_1440":"Anomaly ratio (24 h)","전류고조파최대상_max60":"Max-phase THD, max (60 min)",
"전류고조파평균_max10":"Mean THD, max (10 min)","전류고조파최대상":"Max-phase THD (now)","전류고조파최대상_max30":"Max-phase THD, max (30 min)",
"이상이력없음":"No anomaly history flag","전류고조파_q90_1440":"THD 90th percentile (24 h)","전류평균_max60":"Mean current, max (60 min)","시":"Hour of day"}
ex={"이상비율_1440","전류고조파최대상_max60","전류고조파최대상","전류고조파최대상_max30","전류고조파_q90_1440"}
sv=sv.iloc[::-1]; fig,ax=plt.subplots(figsize=(W,2.3))
ax.barh([nm[f] for f in sv.feature],sv.share*100,color=[BLUE if f in ex else RULE for f in sv.feature],height=0.7)
for yy,q in enumerate(sv.share*100): ax.text(q+0.3,yy,f"{q:.1f}",va="center",fontsize=5.8)
ax.set_xlabel("Share of mean |SHAP| (%)"); ax.grid(axis="y",visible=False)
ax.legend(handles=[Patch(color=RULE,label="Base feature"),Patch(color=BLUE,label="Added feature")],loc="lower right"); save(fig,"fig22")

# 그림 23 (5-12): SHAP 의존도
dp=pd.read_parquet(R+'shap_dependence_sample.parquet')
fig,ax=plt.subplots(figsize=(W,2.0))
for t,c,l in [(0,BLUE,"No onset in 30 min"),(1,ORANGE,"Onset in 30 min")]:
    r=dp[dp.target==t]; ax.scatter(r.value,r.shap,s=2,alpha=0.35 if t==0 else 0.6,color=c,label=l,linewidths=0)
ax.axhline(0,color=MUTED,lw=0.6); ax.set_xscale("symlog",linthresh=10); ax.set_xlim(left=0)
ax.annotate("No history",(dp.value.max(),0),xytext=(-6,8),textcoords="offset points",ha="right",fontsize=5.8,color=MUTED)
ax.set_xlabel("Minutes since last anomaly (symlog)"); ax.set_ylabel("SHAP (log-odds)"); ax.legend(loc="upper right",markerscale=4); save(fig,"fig23")

# 그림 24 (5-13): 경보 대기 시간
al_=rd("extra_fac_v2cmp_alarm.csv")
fig,ax=plt.subplots(figsize=(W,2.3))
for m,c in MODELS:
    dd=al_[al_.model==m].sort_values("wait_min"); ax.plot(dd.false_alarm_per_day,dd.event_recall,"-o",color=c,ms=2.8,mec="white",mew=0.5,label=MN[m])
    if m in ("규칙",PE):
        for _,row in dd.iterrows(): ax.annotate(f"{int(row.wait_min)}",(row.false_alarm_per_day,row.event_recall),xytext=(2,3),textcoords="offset points",fontsize=5.4)
ax.set_xscale("log"); ax.set_xticks([0.2,0.5,1,2,5,10,20],["0.2","0.5","1","2","5","10","20"]); ax.set_ylim(0,1)
ax.set_xlabel("False alarms per unit per day (log)"); ax.set_ylabel("Event recall"); ax.legend(loc="lower right",fontsize=5.4); save(fig,"fig24")

# 그림 25 (5-14): 경보 예시
th=rd('extra_fac_v2cmp_thresholds.csv'); th=th[th.column=='score_2'].set_index('fold').threshold
ev=pd.Timestamp('2020-12-17 21:42'); t0,t1=ev-pd.Timedelta('70min'),ev+pd.Timedelta('15min')
d=sc[(sc.dev==241)&(sc.time>=t0)&(sc.time<ev)].sort_values('time'); thr=th[d.fold.iloc[0]]
cl=pd.read_parquet(f'{D}cleaned/Validation/{PUMP}/241.parquet').loc[t0:t1]; al=ev-pd.Timedelta('24min')
fig,(a1,a2)=plt.subplots(2,1,figsize=(W,2.5),sharex=True,gridspec_kw={"height_ratios":[1,1.2],"hspace":0.12})
a1.plot(cl.index,cl[H],color=INK,lw=0.9); a1.axhline(30,color=RED,ls="--",lw=0.7); a1.text(t0,31,"Caution boundary (30)",color=RED,fontsize=5.8,va="bottom"); a1.set_ylabel("Mean current THD (%)")
a2.plot(d.time,d.score_2,color=BLUE,lw=1.0,label="Model score"); a2.axhline(thr,color=MUTED,ls="--",lw=0.7,label=f"Alarm threshold ({thr:.2f})"); a2.set_ylim(0,1); a2.set_ylabel("Score")
for a in (a1,a2): a.axvline(al,color=ORANGE,lw=1.0); a.axvline(ev,color=RED,lw=1.0)
a2.text(al,0.05," Alarm 21:18",color=ORANGE,fontsize=5.8); a2.text(ev,0.05," Onset 21:42",color=RED,fontsize=5.8)
a2.legend(loc="upper left"); a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); save(fig,"fig25")

# 그림 26 (5-15): 임계값 재보정
rc=rd("recal_v2cmp.csv"); rc=rc[(rc.model==PE)&(rc.days==1)]
short={"기존":"Original","점수 95% 분위 상한":"95%","점수 98% 분위 상한":"98%","점수 99% 분위 상한":"99%","라벨 사용 F1":"Label F1"}
fig,ax=plt.subplots(figsize=(W,2.1))
for wm,c,dx,ha in [(1,BLUE,4,"left"),(2,ORANGE,-4,"right")]:
    dd=rc[rc.wait_min==wm].set_index("method").loc[list(short)]
    ax.plot(dd.false_alarm_per_day,dd.event_recall,"o",ms=3.5,color=c,mec="white",mew=0.6,label=f"Alarm after {wm} consecutive min")
    for m,row in dd.iterrows(): ax.annotate(short[m],(row.false_alarm_per_day,row.event_recall),xytext=(dx,2),textcoords="offset points",fontsize=5.4,ha=ha)
ax.set_xlim(-0.6,8.5); ax.set_ylim(0,1); ax.set_xlabel("False alarms per unit per day"); ax.set_ylabel("Event recall"); ax.legend(loc="lower right",fontsize=5.6); save(fig,"fig26")
print("done")
