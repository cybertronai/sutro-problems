"""Accelerate numerical checks of independent image and class iterations.

Each worker executes the original scalar IL body with private scratch. Only
the iteration's disjoint output slice is merged. Other written scratch is
restored from the final iteration, matching serial observable memory. No
reductions inside an image or a class are reassociated. These host-private
copies are NOT part of the separately scored serial spatial schedule.
"""
from pathlib import Path
import re,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'shared'))
import il_compiler as C
from affine import Program
ORIGINAL=C.emit_cpp


def emit(document):
    program=Program(document)
    cfg=document['metadata']['config']
    mapping={};counter=0
    def visit(body,depth=0):
        nonlocal counter
        for node in body:
            if 'loop' not in node:continue
            ident='v'+str(counter);counter+=1
            if depth==0:mapping[ident]=node
            visit(node['body'],depth+1)
    visit(document['body'])
    def writes(body):
        result=set()
        for n in body:
            if 'loop' in n:result |= writes(n['body'])
            else:
                assert n['op'] not in ('recv','send')
                result.add(n['dst']['region'])
        return result
    lines=ORIGINAL(document).splitlines();out=[];pos=0;changed=[]
    while pos<len(lines):
        match=re.match(r'^  for \(int64_t (v\d+) =',lines[pos])
        node=mapping.get(match[1]) if match else None
        if not node or node['loop'] not in ('f_fn','c'):
            out.append(lines[pos]);pos+=1;continue
        ident=match[1];end=pos;depth=0
        while True:
            depth+=lines[end].count('{')-lines[end].count('}')
            end+=1
            if depth==0:break
        touched=writes(node['body'])
        if node['loop']=='f_fn':
            outputs=touched & {'x','q'}
            assert len(outputs)==1
            output=next(iter(outputs));base,_=program.regions[output]
            merge=f'std::memcpy(shared_m+{base}ULL+{ident}*{cfg["features"]}ULL, m+{base}ULL+{ident}*{cfg["features"]}ULL, {4*cfg["features"]}ULL);'
        else:
            output='sc';base,_=program.regions[output]
            merge=f'for (uint64_t q=0; q<{cfg["Q"]}ULL; ++q) shared_m[{base}ULL+q*10+{ident}]=m[{base}ULL+q*10+{ident}];'
        assert output in touched
        workers=min(10,node['count'])
        last=node['start']+node['count']-1
        out += ['  {', f'    std::fprintf(stderr,"parallel {node["loop"]}\\n"); std::fflush(stderr);', '    uint32_t* shared_m=m;',
            f'    std::vector<std::vector<uint32_t>> private_m({workers});',
            '    int final_worker=-1;',f'    #pragma omp parallel num_threads({workers})',
            '    {','      const int tid=omp_get_thread_num();',
            '      fenv_t worker_env; std::fegetenv(&worker_env); std::fesetround(FE_TONEAREST);',
            '#if defined(__SSE__)',
            '      unsigned worker_mxcsr=_mm_getcsr(); _mm_setcsr(worker_mxcsr & ~(unsigned(1<<15)|unsigned(1<<6)));',
            '#endif',
            f'      private_m[tid].resize({program.words+1}ULL);',
            f'      std::memcpy(private_m[tid].data(),shared_m,{4*(program.words+1)}ULL);',
            '      uint32_t* m=private_m[tid].data();',
            '      #pragma omp barrier',
            '      #pragma omp for schedule(static)']
        out += ['    '+line for line in lines[pos:end-1]]
        out += ['        '+merge,f'        if ({ident}=={last}LL) final_worker=tid;','    '+lines[end-1],
            '#if defined(__SSE__)',
            '      _mm_setcsr(worker_mxcsr);','#endif',
            '      std::fesetenv(&worker_env);','    }']
        for region in sorted(touched-{output}):
            base,size=program.regions[region]
            out.append(f'    std::memcpy(shared_m+{base}ULL,private_m[final_worker].data()+{base}ULL,{4*size}ULL);')
        out.append('  }');changed.append({'loop':node['loop'],'output':output,'private_regions':sorted(touched-{output}),'threads':workers})
        pos=end
    assert len(changed)==3,changed
    return '#include <cstdio>\n#include <vector>\n#include <omp.h>\n'+'\n'.join(out)+'\n'


def install():
    C.emit_cpp=emit
    if '-fopenmp' not in C.FLAGS:C.FLAGS.append('-fopenmp')


if __name__=='__main__':
    install()
    import execute_candidate
    execute_candidate.main()
