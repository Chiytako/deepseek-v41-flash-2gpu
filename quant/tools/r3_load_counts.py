import pickle, zipfile, struct, sys, json
class Stub:
    def __init__(self,*a,**k): self.a=a; self.state=None
    def __setstate__(self,s): self.state=s
    def __call__(self,*a,**k): return Stub(*a)
class Arr(Stub): pass
class DT(Stub): pass
def reconstruct(cls, shape, dtype): return Arr()
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name=='_reconstruct': return reconstruct
        if name=='ndarray': return Arr
        if name=='dtype': return lambda *a,**k: DT(*a)
        if mod=='_codecs' and name=='encode': return lambda s,enc='latin1': s.encode(enc)
        if name=='scalar': return lambda *a: ('scalar',a)
        return lambda *a,**k: ('obj',mod,name)
    def persistent_load(self, pid): return ('storage',)
z=zipfile.ZipFile(sys.argv[1]); n=[x for x in z.namelist() if x.endswith('data.pkl')][0]
obj=U(z.open(n)).load()
counts=obj['counts']
out={}
for k,v in counts.items():
    ver,shape,dt,fortran,raw=v.state
    vals=struct.unpack('<%dq'%(len(raw)//8), raw)
    out[int(k)]=list(vals)
json.dump(out,open(sys.argv[2],'w'))
print(len(out), sum(out[0]), len(out[0]))
