"""Lector de GeoTIFF float32 con deflate + predictor 3 (sin imagecodecs)."""
import tifffile, zlib, numpy as np
def _undo_fp(raw, w, bps=4):
    a=np.frombuffer(raw,dtype=np.uint8).reshape(-1, w*bps)
    a=np.cumsum(a,axis=1,dtype=np.uint8)            # deshace delta de bytes
    a=a.reshape(-1,bps,w).transpose(0,2,1)          # bytes MSB primero
    return a.copy().view('>f4').reshape(-1,w).astype(np.float32)
def lzw_decode(data):
    out=bytearray(); bits=0; nbits=0; width=9; table=[bytes([i]) for i in range(256)]+[b'',b'']; prev=None
    for byte in data:
        bits=(bits<<8)|byte; nbits+=8
        while nbits>=width:
            nbits-=width; code=(bits>>nbits)&((1<<width)-1)
            if code==256: table=table[:258]; width=9; prev=None; continue
            if code==257: return bytes(out)
            if prev is None: entry=table[code]
            else:
                entry=table[code] if code<len(table) else prev+prev[:1]
                table.append(prev+entry[:1])
            out+=entry; prev=entry
            if len(table)+1>=(1<<width) and width<12: width+=1
    return bytes(out)
def read(path):
    t=tifffile.TiffFile(path); p=t.pages[0]
    H,W=p.shape[:2]; pred=p.predictor; fh=t.filehandle
    out=np.empty((H,W),np.float32)
    def dec(off,cnt,w):
        fh.seek(off); raw=fh.read(cnt)
        if p.compression in (8,32946): raw=zlib.decompress(raw)
        elif p.compression==5: raw=lzw_decode(raw)
        if pred==3: return _undo_fp(raw,w)
        a=np.frombuffer(raw,dtype=p.dtype).reshape(-1,w).astype(np.float32)
        return np.cumsum(a,axis=1) if pred==2 else a
    if p.is_tiled:
        th,tw=p.tilelength,p.tilewidth; nx=-(-W//tw); k=0
        for off,cnt in zip(p.dataoffsets,p.databytecounts):
            r,c=divmod(k,nx); k+=1; tile=dec(off,cnt,tw)
            hh=min(th,H-r*th); ww=min(tw,W-c*tw); out[r*th:r*th+hh,c*tw:c*tw+ww]=tile[:hh,:ww]
    else:
        rps=p.rowsperstrip; r=0
        for off,cnt in zip(p.dataoffsets,p.databytecounts):
            s=dec(off,cnt,W); out[r:r+s.shape[0]]=s[:H-r]; r+=s.shape[0]
    sx,sy=p.tags[33550].value[:2]; tp=p.tags[33922].value
    nod=p.tags.get(42113); nod=float(nod.value) if nod else None
    if nod is not None: out[out==nod]=np.nan
    return out, (tp[3]-tp[0]*sx, sx, tp[4]+tp[1]*sy, sy)   # x0, dx, y0, dy
def sample(arr, gt, x, y):
    x0,dx,y0,dy=gt; c=np.floor((np.asarray(x)-x0)/dx).astype(int); r=np.floor((y0-np.asarray(y))/dy).astype(int)
    ok=(r>=0)&(r<arr.shape[0])&(c>=0)&(c<arr.shape[1]); v=np.full(len(c),np.nan); v[ok]=arr[r[ok],c[ok]]; return v
def lcc6372(lon,lat):
    """EPSG:6372 Mexico ITRF2008 / LCC (GRS80)."""
    a=6378137.0; f=1/298.257222101; e=np.sqrt(2*f-f*f)
    p1,p2,p0,l0=np.radians([17.5,29.5,12.0,-102.0]); x0,y0=2500000.0,0.0
    m=lambda p: np.cos(p)/np.sqrt(1-(e*np.sin(p))**2)
    t=lambda p: np.tan(np.pi/4-p/2)/((1-e*np.sin(p))/(1+e*np.sin(p)))**(e/2)
    n=(np.log(m(p1))-np.log(m(p2)))/(np.log(t(p1))-np.log(t(p2))); F=m(p1)/(n*t(p1)**n)
    r0=a*F*t(p0)**n; ph=np.radians(lat); r=a*F*t(ph)**n; th=n*(np.radians(lon)-l0)
    return x0+r*np.sin(th), y0+r0-r*np.cos(th)
