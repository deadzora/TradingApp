
# Minimal intraday portfolio DCA backtest (15m, ~60d Yahoo window)
import pandas as pd, numpy as np, yfinance as yf
def rsi(s, n=14):
    d=s.diff(); g=(d.where(d>0,0)).rolling(n).mean(); l=(-d.where(d<0,0)).rolling(n).mean(); rs=g/(l+1e-9); return 100-100/(1+rs)
def atr(h,l,c,n=14):
    pc=c.shift(1); tr=pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()],axis=1).max(axis=1); return tr.rolling(n).mean()
def ind(df):
    out={}; out['ma_50']=df['close'].rolling(50).mean(); out['ma_200']=df['close'].rolling(200).mean(); out['rsi_14']=rsi(df['close']); out['atr_14']=atr(df['high'],df['low'],df['close']); out['ret_20']=df['close'].pct_change(20); out['vol_z']=(df['volume']-df['volume'].rolling(20).mean())/(df['volume'].rolling(20).std()+1e-9); return pd.DataFrame(out,index=df.index)
def score(row):
    mom=0; 
    if pd.notna(row.get('ma_50')) and pd.notna(row.get('ma_200')): mom += (5 if row['close']>row['ma_50'] else 0)+(5 if row['close']>row['ma_200'] else 0)
    mom += 5 if (row.get('ret_20') or 0)>0 else 0; r=row.get('rsi_14',50) or 50; mom += 5 if 40<=r<=70 else 0
    liq=10 if (row.get('vol_z') or 0)>0 else 0; liq+=10; return 40*(mom/20.0)+20*(liq/20.0)+20*0.5+20*0.5
def mrsig(df): r=df['close'].pct_change(); z=(r-r.rolling(20).mean())/(r.rolling(20).std()+1e-9); return z.iloc[-1]<-2
def momsig(df): return df['close'].iloc[-1]>=df['close'].rolling(20).max().iloc[-2]
def fetch15(symbol, start, end):
    d=yf.download(symbol, start=start, end=end, interval='15m', progress=False, threads=False)
    if d is None or d.empty: return None
    d=d.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})[['open','high','low','close','volume']].dropna(); return d
def run():
    syms=['SPY','QQQ','AAPL','NVDA','MSFT']; start=(pd.Timestamp.utcnow()-pd.Timedelta(days=60)).strftime('%Y-%m-%d'); end=pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    frames={s:fetch15(s,start,end) for s in syms}; frames={k:v for k,v in frames.items() if v is not None and len(v)>100}
    if not frames: print('No data'); return
    idx=sorted(set().union(*[f.index for f in frames.values()])); equity=100.0; last=idx[0]; trades=[]; pos={}; dca_amt=100; dca_min=15*24*14
    for t in idx:
        if (t-last).total_seconds()>=dca_min*60: equity+=dca_amt; last=t
        # exits
        for s in list(pos.keys()):
            if t not in frames[s].index: continue
            px=float(frames[s].loc[t,'close']); q,e=pos[s]; stop=e*0.99; take=e*1.02
            if px<=stop or px>=take:
                equity+=q* (stop if px<=stop else take); trades.append({'ts':t,'symbol':s,'side':'sell','qty':q}); del pos[s]
        snaps=[]
        for s,df in frames.items():
            if t not in df.index: continue
            sub=df.loc[:t].tail(220); feat=ind(sub).tail(1).iloc[0].to_dict(); feat['close']=float(sub['close'].iloc[-1]); snaps.append((s,score(feat),feat))
        if not snaps: continue
        snaps=sorted(snaps,key=lambda x:x[1],reverse=True)[:3]
        for s,sc,feat in snaps:
            if s in pos: continue
            price=feat['close']; sub=frames[s].loc[:t].tail(220); fire=momsig(sub) or mrsig(sub)
            if not fire or price<=0: continue
            qty=int((equity/len(snaps))/max(price,1)*0.1); 
            if qty<1 or qty*price>equity: continue
            equity-=qty*price; pos[s]=(qty,price); trades.append({'ts':t,'symbol':s,'side':'buy','qty':qty})
    for s,(q,e) in pos.items():
        last_px=float(frames[s]['close'].iloc[-1]); equity+=q*last_px; trades.append({'ts':frames[s].index[-1],'symbol':s,'side':'sell','qty':q})
    pd.DataFrame(trades).to_csv('intraday_portfolio_trades.csv', index=False); print(f'Final equity: {equity:.2f}')
if __name__=='__main__': run()
