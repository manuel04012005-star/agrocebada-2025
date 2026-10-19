import struct, numpy as np, pandas as pd
P='/home/claude/datos/Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO'
def read_dbf():
    d=open(P+'.dbf','rb').read(); n,hl,rl=struct.unpack('<IHH',d[4:12]); f=[]; i=32
    while d[i]!=0x0D: f.append((d[i:i+11].split(b'\0')[0].decode('utf-8','ignore'),d[i+16])); i+=32
    rows=[]; o=hl
    for _ in range(n):
        r=d[o+1:o+rl]; o+=rl; pos=0; row=[]
        for _,L in f: row.append(r[pos:pos+L].decode('utf-8','ignore').strip()); pos+=L
        rows.append(row)
    df=pd.DataFrame(rows,columns=['Cultivo','ID_POLIGONO','area_ha','Municipio','Estado','CONJUNTO'])
    df['area_ha']=df.area_ha.astype(float); return df
def read_shp():
    b=open(P+'.shp','rb').read(); o=100; geoms=[]
    while o<len(b):
        clen=struct.unpack('>i',b[o+4:o+8])[0]*2; rec=b[o+8:o+8+clen]; o+=8+clen
        st=struct.unpack('<i',rec[:4])[0]
        npart,npt=struct.unpack('<2i',rec[36:44]); parts=list(struct.unpack(f'<{npart}i',rec[44:44+4*npart]))
        pts=np.frombuffer(rec[44+4*npart:44+4*npart+16*npt],dtype='<f8').reshape(-1,2)
        parts.append(npt); geoms.append([pts[parts[k]:parts[k+1]] for k in range(npart)])
    return geoms
def load():
    df=read_dbf(); g=read_shp(); df['rings']=g
    df['lon']=[np.concatenate(r)[:,0].mean() for r in g]; df['lat']=[np.concatenate(r)[:,1].mean() for r in g]
    return df
