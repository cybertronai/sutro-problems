// Numerical specialization of the affine grid program. Every dependency-chain
// reduction follows ascending grid indices with separate FP32 multiply/add.
// Build with -O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math.
#include <algorithm>
#include <array>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {
constexpr int D=81, C=10, P=16;
using Vec=std::vector<float>;
using Callback=void(*)(const char*,const float*,int64_t,int64_t,void*);
thread_local std::string last_error;
float root(float x) {
    float y=x+1e-6f;
    for(int it=0;it<6;++it) { float q=x/y; q=y+q; y=.5f*q; }
    return y;
}
float asin_series(float x) {
    float u=x*x;
    float t=float(35.0/1152.0)*u; t=t+float(5.0/112.0);
    t=t*u; t=t+float(3.0/40.0); t=t*u; t=t+float(1.0/6.0);
    t=t*u; t=t+1.f; return t*x;
}
float transform(float x) {
    x=x<0.f?0.f:x; x=1.f<x?1.f:x;
    float s=root(x), a=asin_series(s), s2=s*s;
    s2=1.f-s2; s2=root(s2);
    float b=asin_series(s2); b=float(3.14159265358979323846/2.0)-b;
    return .7f<s?b:a;
}
float exp_neg(float x) {
    float t=x*float(-1.0/256.0);
    float u=t*float(1.0/24.0); u=u+float(1.0/6.0);
    u=u*t; u=u+.5f; u=u*t; u=u+1.f; u=u*t; float e=u+1.f;
    for(int i=0;i<8;++i) e=e*e;
    return e;
}
struct Run {
    Callback cb; void* context; bool verbose;
    std::chrono::steady_clock::time_point last=std::chrono::steady_clock::now();
    void snapshot(const char* name,const Vec& x,int64_t rows,int64_t cols) {
        if(cb) cb(name,x.data(),rows,cols,context);
        if(verbose) {
            auto now=std::chrono::steady_clock::now();
            std::fprintf(stderr,"%s %.3f s\n",name,std::chrono::duration<double>(now-last).count());
            std::fflush(stderr); last=now;
        }
    }
};
Vec pixels(const float* x,int n) {
    Vec u(int64_t(n)*D);
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<int64_t(n)*D;++i) u[i]=transform(x[i]);
    return u;
}
Vec features(const Vec& u,int n,int F,const float* w,const float* bias) {
    int nf=F*9; Vec out(int64_t(n)*nf,0.f);
    #pragma omp parallel for schedule(static)
    for(int row=0;row<n;++row) {
        std::array<float,121> pad{};
        for(int y=0;y<9;++y) for(int x=0;x<9;++x) pad[(y+1)*11+x+1]=u[int64_t(row)*D+y*9+x];
        for(int f=0;f<F;++f) for(int cy=0;cy<3;++cy) for(int cx=0;cx<3;++cx) {
            float pooled=0.f;
            for(int iy=0;iy<3;++iy) for(int ix=0;ix<3;++ix) {
                float acc=bias[f];
                for(int dy=0;dy<3;++dy) for(int dx=0;dx<3;++dx) {
                    float t=w[f*9+dy*3+dx]*pad[dy*11+dx+cy*33+iy*11+cx*3+ix];
                    acc=acc+t;
                }
                float t=acc<0.f?0.f:acc; t=t*float(1.0/9.0); pooled=pooled+t;
            }
            out[int64_t(row)*nf+f*9+cy*3+cx]=pooled;
        }
    }
    return out;
}
Vec center(const Vec& phi,int n,int nf,Vec& mean) {
    mean.assign(nf,0.f);
    for(int i=0;i<n;++i) {
        #pragma omp simd
        for(int f=0;f<nf;++f) mean[f]=mean[f]+phi[int64_t(i)*nf+f];
    }
    float invn=float(1.0/n);
    for(int f=0;f<nf;++f) mean[f]=mean[f]*invn;
    Vec centered(phi.size());
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i) {
        #pragma omp simd
        for(int f=0;f<nf;++f) centered[int64_t(i)*nf+f]=phi[int64_t(i)*nf+f]-mean[f];
    }
    return centered;
}
Vec gram(const Vec& phi,int n,int nf) {
    constexpr int BS=64;
    Vec G(int64_t(nf)*nf,0.f);
    std::vector<std::pair<int,int>> tiles;
    for(int bi=0;bi<nf;bi+=BS) for(int bj=bi;bj<nf;bj+=BS) tiles.push_back({bi,bj});
    #pragma omp parallel for schedule(dynamic,1)
    for(size_t tile=0;tile<tiles.size();++tile) {
        int bi=tiles[tile].first,bj=tiles[tile].second;
        alignas(64) float acc[BS][BS]{};
        int ni=std::min(BS,nf-bi),nj=std::min(BS,nf-bj);
        for(int k=0;k<n;++k) {
            const float* row=phi.data()+int64_t(k)*nf;
            for(int i=0;i<ni;++i) {
                float a=row[bi+i];
                #pragma omp simd
                for(int j=0;j<nj;++j) { float t=a*row[bj+j]; acc[i][j]=acc[i][j]+t; }
            }
        }
        for(int i=0;i<ni;++i) for(int j=0;j<nj;++j) {
            G[int64_t(bi+i)*nf+bj+j]=acc[i][j];
            G[int64_t(bj+j)*nf+bi+i]=acc[i][j];
        }
    }
    return G;
}
Vec target(const int64_t* y,int n) {
    Vec Y(int64_t(n)*C,-1.f);
    for(int i=0;i<n;++i) { if(y[i]<0||y[i]>=C) throw std::runtime_error("invalid label"); Y[int64_t(i)*C+y[i]]=1.f; }
    return Y;
}
Vec rhs(const Vec& phi,const Vec& Y,int n,int nf) {
    Vec B(int64_t(nf)*C,0.f);
    #pragma omp parallel for schedule(static)
    for(int f=0;f<nf;++f) {
        alignas(64) float acc[P]{};
        for(int i=0;i<n;++i) {
            float v=phi[int64_t(i)*nf+f];
            #pragma omp simd
            for(int c=0;c<C;++c) { float t=v*Y[int64_t(i)*C+c]; acc[c]=acc[c]+t; }
        }
        for(int c=0;c<C;++c) B[int64_t(f)*C+c]=acc[c];
    }
    return B;
}
Vec kernel(const Vec& u,int n,const Vec& v,int m,float gamma) {
    constexpr int BS=64;
    Vec vt(int64_t(D)*m),K(int64_t(n)*m);
    for(int d=0;d<D;++d) for(int j=0;j<m;++j) vt[int64_t(d)*m+j]=v[int64_t(j)*D+d];
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i) {
        for(int j0=0;j0<m;j0+=BS) {
            alignas(64) float acc[BS]{}; int size=std::min(BS,m-j0);
            for(int d=0;d<D;++d) {
                float a=u[int64_t(i)*D+d];
                #pragma omp simd
                for(int j=0;j<size;++j) { float t=a-vt[int64_t(d)*m+j0+j]; t=t*t; acc[j]=acc[j]+t; }
            }
            #pragma omp simd
            for(int j=0;j<size;++j) { float t=acc[j]*gamma; K[int64_t(i)*m+j0+j]=exp_neg(t); }
        }
    }
    return K;
}
// Padded class rows permit vectorization only across independent class columns.
Vec cg(const Vec& M,const Vec& B,const Vec& inverse,int n,int T,Run& run,bool kernel_cg) {
    Vec x(int64_t(n)*P,0.f),r(x.size(),0.f),z(x.size(),0.f),p(x.size(),0.f),ap(x.size(),0.f);
    alignas(64) float rz[P]{},rzn[P]{},alpha[P]{};
    for(int i=0;i<n;++i) for(int c=0;c<C;++c) {
        int64_t j=int64_t(i)*P+c; r[j]=B[int64_t(i)*C+c]; z[j]=r[j]*inverse[i]; p[j]=z[j];
        float t=r[j]*z[j]; rz[c]=rz[c]+t;
    }
    for(int it=0;it<T;++it) {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i) {
            alignas(64) float acc[P]{};
            const float* row=M.data()+int64_t(i)*n;
            for(int j=0;j<n;++j) {
                float v=row[j];
                #pragma omp simd
                for(int c=0;c<P;++c) { float t=v*p[int64_t(j)*P+c]; acc[c]=acc[c]+t; }
            }
            for(int c=0;c<P;++c) ap[int64_t(i)*P+c]=acc[c];
        }
        for(int c=0;c<C;++c) alpha[c]=0.f;
        for(int i=0;i<n;++i) {
            #pragma omp simd
            for(int c=0;c<C;++c) { int64_t j=int64_t(i)*P+c; float t=p[j]*ap[j]; alpha[c]=alpha[c]+t; }
        }
        for(int c=0;c<C;++c) alpha[c]=rz[c]/alpha[c];
        for(int i=0;i<n;++i) {
            #pragma omp simd
            for(int c=0;c<C;++c) {
                int64_t j=int64_t(i)*P+c; float t=alpha[c]*p[j]; x[j]=x[j]+t;
                t=alpha[c]*ap[j]; r[j]=r[j]-t;
            }
        }
        for(int i=0;i<n;++i) {
            #pragma omp simd
            for(int c=0;c<C;++c) { int64_t j=int64_t(i)*P+c; z[j]=r[j]*inverse[i]; }
        }
        for(int c=0;c<C;++c) rzn[c]=0.f;
        for(int i=0;i<n;++i) {
            #pragma omp simd
            for(int c=0;c<C;++c) { int64_t j=int64_t(i)*P+c; float t=r[j]*z[j]; rzn[c]=rzn[c]+t; }
        }
        for(int c=0;c<C;++c) { alpha[c]=rzn[c]/rz[c]; rz[c]=rzn[c]; }
        for(int i=0;i<n;++i) {
            #pragma omp simd
            for(int c=0;c<C;++c) { int64_t j=int64_t(i)*P+c; float t=alpha[c]*p[j]; p[j]=z[j]+t; }
        }
    }
    Vec out(int64_t(n)*C);
    for(int i=0;i<n;++i) for(int c=0;c<C;++c) out[int64_t(i)*C+c]=x[int64_t(i)*P+c];
    if(run.cb) {
        for(auto item : {std::pair<const char*,const Vec*>{kernel_cg?"pk":"p",&p},
                        {kernel_cg?"apk":"ap",&ap},{kernel_cg?"rk":"rr",&r},
                        {kernel_cg?"zk":"zz",&z}}) {
            Vec packed(int64_t(n)*C);
            for(int i=0;i<n;++i) for(int c=0;c<C;++c) packed[int64_t(i)*C+c]=(*item.second)[int64_t(i)*P+c];
            run.snapshot(item.first,packed,n,C);
        }
        if(kernel_cg) {
            Vec scratch(64*64,0.f);
            for(int c=0;c<C;++c) { scratch[c]=rz[c]; scratch[C+c]=rzn[c]; scratch[2*C+c]=alpha[c]; }
            run.snapshot("gb",scratch,64,64);
        }
    }
    return out;
}
Vec scores(const Vec& left,const Vec& right,int q,int width,const Vec* mean=nullptr) {
    Vec out(int64_t(q)*C);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<q;++i) {
        alignas(64) float acc[P]{};
        for(int j=0;j<width;++j) {
            float v=left[int64_t(i)*width+j]; if(mean) v=v-(*mean)[j];
            #pragma omp simd
            for(int c=0;c<C;++c) { float t=v*right[int64_t(j)*C+c]; acc[c]=acc[c]+t; }
        }
        for(int c=0;c<C;++c) out[int64_t(i)*C+c]=acc[c];
    }
    return out;
}
void standardize(Vec& s,int q) {
    for(int i=0;i<q;++i) {
        float mu=0.f; for(int c=0;c<C;++c) mu=mu+s[int64_t(i)*C+c]; mu=mu*.1f;
        float sd=0.f; for(int c=0;c<C;++c) { float t=s[int64_t(i)*C+c]-mu; t=t*t; sd=sd+t; }
        sd=sd*float(1.0/9.0); sd=root(sd);
        for(int c=0;c<C;++c) { int64_t j=int64_t(i)*C+c; s[j]=s[j]-mu; s[j]=s[j]/sd; }
    }
}
}

extern "C" const char* cg_grid_last_error() { return last_error.c_str(); }
extern "C" int cg_grid_run(const float* x,const int64_t* labels,const float* q,
    const float* weights,const float* biases,int N,int Q,int F,int T,
    float gamma,float lam_ridge,float lam_k,int threads,int verbose,
    int64_t* predictions,float* combined_scores,Callback callback,void* context) {
    try {
        if(N<=0||Q<=0||F<=0||T<0) throw std::runtime_error("invalid shape");
        if(threads>0) omp_set_num_threads(threads);
        omp_set_dynamic(0);
        #pragma omp parallel
        {
            std::fesetround(FE_TONEAREST);
            #if defined(__SSE__)
            _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040)); // no flush-to-zero or denormals-are-zero
            #endif
        }
        std::fesetround(FE_TONEAREST);
        #if defined(__SSE__)
        _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040));
        #endif
        Run run{callback,context,verbose!=0}; int nf=9*F;
        Vec u=pixels(x,N); run.snapshot("x",u,N,D);
        Vec phi=features(u,N,F,weights,biases); run.snapshot("Phi",phi,N,nf);
        Vec mean; Vec centered=center(phi,N,nf,mean); run.snapshot("mean",mean,nf,1);
        Vec Y=target(labels,N); run.snapshot("Y",Y,N,C);
        Vec G=gram(centered,N,nf); run.snapshot("G_unregularized",G,nf,nf);
        Vec B=rhs(centered,Y,N,nf); run.snapshot("B",B,nf,C);
        float trace=0.f; for(int i=0;i<nf;++i) trace=trace+G[int64_t(i)*nf+i];
        float ridge=trace*lam_ridge; ridge=ridge*float(1.0/nf);
        Vec inverse(nf); for(int i=0;i<nf;++i) { int64_t j=int64_t(i)*nf+i; G[j]=G[j]+ridge; inverse[i]=1.f/G[j]; }
        run.snapshot("G",G,nf,nf); run.snapshot("diag",inverse,nf,1);
        Vec X=cg(G,B,inverse,nf,T,run,false); run.snapshot("X",X,nf,C);
        Vec K=kernel(u,N,u,N,gamma);
        Vec kinverse(N); for(int i=0;i<N;++i) { int64_t j=int64_t(i)*N+i; K[j]=K[j]+lam_k; kinverse[i]=1.f/K[j]; }
        run.snapshot("K",K,N,N); run.snapshot("diagk",kinverse,N,1);
        Vec A=cg(K,Y,kinverse,N,T,run,true); run.snapshot("A",A,N,C);
        Vec uq=pixels(q,Q); run.snapshot("xq",uq,Q,D);
        Vec phiq=features(uq,Q,F,weights,biases); run.snapshot("PhiQ",phiq,Q,nf);
        Vec sc1=scores(phiq,X,Q,nf,&mean); run.snapshot("sc1_raw",sc1,Q,C);
        Vec KQ=kernel(uq,Q,u,N,gamma); run.snapshot("KQ",KQ,Q,N);
        Vec sc2=scores(KQ,A,Q,N); run.snapshot("sc2_raw",sc2,Q,C);
        standardize(sc1,Q); run.snapshot("sc1_normalized",sc1,Q,C);
        standardize(sc2,Q); run.snapshot("sc2_normalized",sc2,Q,C);
        for(int i=0;i<Q;++i) {
            for(int c=0;c<C;++c) { int64_t j=int64_t(i)*C+c; sc1[j]=sc1[j]+sc2[j]; combined_scores[j]=sc1[j]; }
            float best=sc1[int64_t(i)*C]; int64_t lab=0;
            for(int c=1;c<C;++c) if(best<sc1[int64_t(i)*C+c]) { best=sc1[int64_t(i)*C+c]; lab=c; }
            predictions[i]=lab;
        }
        run.snapshot("sc1",sc1,Q,C); last_error.clear(); return 0;
    } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
