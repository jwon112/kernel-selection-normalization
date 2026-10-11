"""Observation-only selector hooks and chunked float64 arithmetic diagnostics."""
from contextlib import AbstractContextManager
import types
import torch


def cpu_copy(x):
    return x.detach().to('cpu').clone()


class Capture(AbstractContextManager):
    """Clone post-IN before in-place activation; never replace a forward output."""
    def __init__(self, selector):
        self.selector=selector;self.data={};self.handles=[]

    def __enter__(self):
        s=self.selector
        if s.post_norm.affine or s.post_norm.track_running_stats:
            raise ValueError('Audit currently requires recorded affine=False, no running statistics')
        self.had_override='_drop_dense_weights' in s.__dict__
        self.old_override=s.__dict__.get('_drop_dense_weights')
        original=s._drop_dense_weights
        def weights(module, value):
            result=original(value)
            self.data['weights_before']=cpu_copy(value)
            self.data['weights_after']=cpu_copy(result)
            dropped=value.detach().clone()
            index=s.branch_names.index(s._intervention['branch'])
            dropped[:,index]=0
            self.data['mass_native']=cpu_copy(dropped.sum(dim=1))
            self.data['denominator_native']=cpu_copy(dropped.sum(dim=1).clamp_min(1e-8))
            self.data['removed_index']=index
            return result
        s._drop_dense_weights=types.MethodType(weights,s)
        def incoming(module,args):self.data['input']=cpu_copy(args[0])
        def pre_norm(module,args):self.data['z']=cpu_copy(args[0])
        def post_norm(module,args,output):self.data['y']=cpu_copy(output)
        self.handles.append(s.register_forward_pre_hook(incoming))
        self.handles.append(s.post_norm.register_forward_pre_hook(pre_norm))
        self.handles.append(s.post_norm.register_forward_hook(post_norm))
        self.data['branches']={}
        for name,branch in s.branches.items():
            def output_hook(module,args,output,name=name):self.data['branches'][name]=cpu_copy(output)
            self.handles.append(branch.register_forward_hook(output_hook))
        self.data.update(epsilon=float(s.post_norm.eps),branch_names=list(s.branch_names))
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()
        if self.had_override:self.selector._drop_dense_weights=self.old_override
        else:del self.selector._drop_dense_weights


def rms(x):return float(x.square().mean().sqrt())


def relative(error,reference):
    denominator=rms(reference)
    return rms(error)/denominator if denominator>0 else None


def norm(x,eps):
    centered=x-x.mean();v=centered.square().mean()
    if float(v)+eps==0:return None
    return centered/torch.sqrt(v+eps)


def channel_diagnostics(a,b,case_ids):
    """Yield one channel at a time to bound float64 temporary memory."""
    for key in ('input','weights_before'):
        if not torch.equal(a[key],b[key]):raise ValueError('Paired upstream data differ: '+key)
    for name in a['branch_names']:
        if not torch.equal(a['branches'][name],b['branches'][name]):
            raise ValueError('Paired branch tensors differ: '+name)
    eps=a['epsilon'];removed=a['removed_index']
    if eps!=b['epsilon'] or removed!=b['removed_index']:raise ValueError('Paired operator mismatch')
    for n,case in enumerate(case_ids):
        for c in range(a['z'].shape[1]):
            zd=a['z'][n,c].double();zr=b['z'][n,c].double()
            yd=a['y'][n,c].double();yr=b['y'][n,c].double()
            v=float((zd-zd.mean()).square().mean())
            mass=float(a['mass_native'][n,c]);den=float(b['denominator_native'][n,c])
            t=1./den;ratio=((v+eps)/(v+eps/(t*t)))**.5
            weights=a['weights_before'][n,:,c].double().clone();weights[removed]=0
            ideal=torch.zeros_like(zd)
            for k,name in enumerate(a['branch_names']):ideal.add_(a['branches'][name][n,c].double()*weights[k])
            s=float(weights.sum());ideal_scale=1/max(s,1e-8)
            expected_weights=weights*ideal_scale if s>0 else weights
            row={'case_id':case,'channel':c,'variance':v,'epsilon':eps,'variance_over_epsilon':v/eps,
                 'surviving_mass_native':mass,'surviving_mass_float64':s,'zero_mass':mass==0,
                 'below_floor':0<mass<1e-8,'scale':t,'z_dtype':str(a['z'].dtype),'y_dtype':str(a['y'].dtype),
                 'weight_dtype':str(a['weights_before'].dtype),
                 'pre_scale_rmse':rms(zr-t*zd),'pre_scale_relative':relative(zr-t*zd,t*zd),
                 'post_difference_rmse':rms(yr-yd),'post_difference_relative':relative(yr-yd,yd),
                 'finite_epsilon_prediction_rmse':rms(yr-ratio*yd),'finite_epsilon_ratio':ratio,
                 'native_drop_vs_float64_sum_rmse':rms(zd-ideal),
                 'renorm_weight_max_error':float((b['weights_after'][n,:,c].double()-expected_weights).abs().max())}
            for e in (0.,1e-8,1e-5,1e-3):
                label=f'{e:g}'
                x=norm(ideal,e);y=norm(ideal*ideal_scale,e)
                row['replay_difference_rms_eps_'+label]=None if x is None else rms(y-x)
                if x is None:residual=None
                else:
                    vi=float((ideal-ideal.mean()).square().mean())
                    r=((vi+e)/(vi+e/ideal_scale**2))**.5
                    residual=float((y-r*x).abs().max())
                row['replay_identity_max_error_eps_'+label]=residual
            yield row
