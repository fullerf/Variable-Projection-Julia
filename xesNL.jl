module xesNL

export
    #types 
    LineShape, 
    BoundedLineShape,  
    ModelParams, 
    BoundedLinSolver,
    UnboundedLinSolver,
    NLGlobalSolver,
    NLLocalSolver,
    Stats,

    #lineshapes 
    lorentzFun!, 
    voigtFun!, 
    asymmetricVoigtFun!, 
    constOffset!, 
    linearOffset!,
    fixedLine!, 
    gaussianFun!, 
    asymmetricGaussian!,

    #utility functions 
    mapLines!, 
    vecrange 

using NLopt
using Distributions

## Custom Types & Call Overloads

#At the core of this program, we allow the user to specify the basis we fit to the data by a vector of functions
#This vector of functions is evaluated over an axis, resulting in each function's output being stored into the columns of a matrix.
#This requires we use a "higher order function", or a function of functions.  Julia sucks at this.  But Julia, for reasons
#I cannot explain, can overload types to be evaluated like functions and then functions of these callable types work great.  So we
#need to wrap the functions we want to evaluate to form the basis in a type and then make the type callable.  It turns out this
#is convenient because it lets us also store a fixed state for the function and this state then no longer needs to be specified
#in the type's argument list.

immutable LineShape{T<:Base.LinAlg.BlasFloat}
    f::Function
    fixed::Array{T,1}
    function LineShape(f::Function,fixed::Array{T,1})
        return new(f,fixed)
    end
    function LineShape(f::Function)
        return new(f,T[])
    end
end
call{T<:Base.LinAlg.BlasFloat}(f::LineShape{T},p::Ptr{T},i::Int,x::Array{T,1},β::Array{T,1}) = f.f(p,i,x,β,f.fixed)

immutable BoundedLineShape{T<:Base.LinAlg.BlasFloat}
    f::LineShape{T}
    ub::Array{T,1}
    lb::Array{T,1}
    function BoundedLineShape(f,ub,lb)
        @assert length(lb)==length(ub)
        @assert all(lb.<ub)
        return new(f,ub,lb)
    end
end
call{T<:Base.LinAlg.BlasFloat}(f::BoundedLineShape{T},p::Ptr{T},i::Int,x::Array{T,1},β::Array{T,1}) = f.f(p,i,x,β)

immutable ModelParams{T<:Base.LinAlg.BlasFloat}
    f::Array{BoundedLineShape{T},1}
    βlb::Array{T,1}
    βub::Array{T,1}
    αlb::Array{T,1}
    αub::Array{T,1}
    αlen::Int
    βlen::Int
    xlen::Int
    axis::Array{T,1}
    y::Array{T,1}
    function ModelParams(f::Array{BoundedLineShape{T},1},αlb::Array{T,1},αub::Array{T,1},axis::Array{T,1},y::Array{T,1})
        αlen = length(f)
        @assert length(αlb)==length(αub)
        if length(αlb)!=0
            @assert all(αlb.<αub)
            @assert length(αlb)==αlen
        end
        βlb = copy(f[1].lb)
        βub = copy(f[1].ub)
        for k=2:αlen
            append!(βlb, f[k].lb)
            append!(βub, f[k].ub)
        end
        @assert size(y,1)==length(axis)
        return new(f,βlb,βub,αlb,αub,αlen,length(βub),length(axis),axis,y)
    end
end

#LinSystem is a container type which allocates memory space for a linear solver method
#this type should not be exported to the user; it's part of the guts
type LinSystem{T<:Base.LinAlg.BlasFloat}
    params::ModelParams{T}
    y::Array{T,1}           #holds the target of the fit
    m::Array{T,2}           #holds the parametric basis
    β::Array{T,1}           #holds the nonlinear parameters
    α::Array{T,1}           #holds the linear parameters
    residual::Array{T,1}    #holds the residual after the fit
    t::Array{T,1}           #a temporary vector for storage
    rss::T                  #the residual sum of squares
    calls::Int
    function LinSystem(params)
        #we make a copy of y here in case we want to mutate this
        #never you mind that mutable objects inside an immutable object can
        #still be mutated by pointers...
        y = copy(params.y)
        m = zeros(T,params.xlen,params.αlen)
        α = zeros(T,params.αlen)
        β = (params.βlb + params.βub)/2
        rss = zero(T)
        residual = zeros(T,params.xlen)
        t = zeros(T,params.xlen)
        calls = 0
        return new(params,y,m,β,α,residual,t,rss,calls)
    end
end

#We allow for multiple kinds of linear solvers.  LinSolvers should have the following traits:
#   1. Each solver is given its own type which are subtype to the abstract LinSolver type
#   2. Solvers are exported to the user.
#   3. Instantiation of a solver type will allocate its own memory bag (LinSystem),
#      based on the ModelParam structure passed to it.
#   4. LinSolvers have an overloaded call function accepting a single parameter:
#      β::Vector{T} and the call should return void, updating its LinSystem fields: α, rss, and m.
#

abstract LinSolver{T<:Base.LinAlg.BlasFloat}

type BoundedLinSolver{T<:Base.LinAlg.BlasFloat} <: LinSolver{T}
    ls::LinSystem{T}
    o::Opt
    function BoundedLinSolver(params::ModelParams{T})
        ls = LinSystem{T}(params)
        o = Opt(:LN_BOBYQA,ls.params.αlen)
        lower_bounds!(o,ls.params.αlb)
        upper_bounds!(o,ls.params.αub)
        xtol_rel!(o,1.5e-8)
        this = new(ls,o)
        obj(α::Array{T,1},g=T[]) = linObjective(α,this.ls)
        min_objective!(o,obj)
        return this
    end
end

function call{T<:Base.LinAlg.BlasFloat}(bls::BoundedLinSolver{T}, β::Array{T,1})
    initα(bls.ls)
    updateBasis(bls.ls,β)
    optimize!(bls.o,bls.ls.α)
    return nothing
end

immutable GenInv{T<:Base.LinAlg.BlasFloat}
    Φinv::Array{T,2}
    Φ::Array{T,2}
    q::Array{T,2}
    r::Array{T,2}
    p::Array{Int64,1}
    ip::Array{Int64,1}
    d::Array{T,1}
    L::Int
    K::Int
    function GenInv(Φinput::Array{T,2})
        Φ = copy(Φinput)
        (L,K) = size(Φ)
        Φinv = zeros(T,(K,L))
        q = zeros(T,(L,K))
        r = zeros(T,(K,K))
        d = zeros(T,K)
        p = zeros(T,K)
        ip = zeros(T,K)
        swap = zeros(T,(1,L))
        return new(Φinv,Φ,q,r,p,ip,d,L,K)
    end
end

function call{T<:Base.LinAlg.BlasFloat}(gi::GenInv{T},Φ)
    (L,K) = size(Φ)
    @assert L==gi.L
    @assert K==gi.K
    qr!(Φ,gi.q,gi.r,gi.p)
    absdiag!(gi.d,gi.r)
    k = numSigElems(gi.d)
    sr = sub(gi.r,1:k,1:k)
    Base.LinAlg.LAPACK.trtri!('U','N',sr) #r is now filled with rinv
    fill!(gi.Φinv,zero(T))
    sΦinv = sub(gi.Φinv,1:k,1:L) 
    sq = sub(gi.q,1:L,1:k)
    #the following: rinv*q', storing into Φinv
    Base.LinAlg.BLAS.gemm!('N','T',one(T),sr,sq,zero(T),sΦinv)
    rangefill!(gi.ip)
    Base.ipermute!!(gi.ip,gi.p) #this screws up gi.p.  But gi.ip is all we want externally.
    return gi.Φinv
end


type UnboundedLinSolver{T<:Base.LinAlg.BlasFloat} <: LinSolver{T}
    ls::LinSystem{T}
    gi::GenInv{T}
    function UnboundedLinSolver(params::ModelParams{T})
        ls = LinSystem{T}(params)
        gi = GenInv{T}(ls.m)
        return new(ls,gi)
    end
end

function call{T<:Base.LinAlg.BlasFloat}(uls::UnboundedLinSolver{T}, β::Array{T,1})
    updateBasis(uls.ls,β)
    Base.LinAlg.BLAS.blascopy!(length(uls.ls.m),uls.ls.m,stride(uls.ls.m,1),uls.gi.Φ,stride(uls.gi.Φ,1))
    uls.gi(uls.gi.Φ) #Φ is now corrupted, this is why we preserved ls.m by copying
    Base.LinAlg.BLAS.gemv!('N',one(T),uls.gi.Φinv,uls.ls.y,zero(T),uls.ls.α) #α = Φinv*y, but we need to permute α to match ls.m ordering
    Base.permute!!(uls.ls.α,uls.gi.ip) #the ip vector is now lost.
    Base.LinAlg.BLAS.blascopy!(length(uls.ls.y),uls.ls.y,stride(uls.ls.y,1),uls.ls.residual,stride(uls.ls.residual,1))
    Base.LinAlg.BLAS.gemv!('N',one(T),uls.ls.m,uls.ls.α,-one(T),uls.ls.residual)
    uls.ls.rss = Base.LinAlg.BLAS.dot(length(uls.ls.residual),uls.ls.residual,stride(uls.ls.residual,1),uls.ls.residual,stride(uls.ls.residual,1))
    return nothing
end

## Here be the nonlinear solvers.  I make one for global solutions and one for local solutions
# the difference is just whether or not a global algorithm from NLopt is added on or not.  
# For any time you're unsure of a good solution you'll use a global algorithm.  After you have
# a solution from a global optimizer you can "polish" it with the local solver, which is set
# to use higher precision.  With bootstrapping, the local solver will also let you explore the
# local minimum you found with the global solver.

abstract NLSolver{T<:Base.LinAlg.BlasFloat}

type NLGlobalSolver{T<:Base.LinAlg.BlasFloat} <: NLSolver{T}
    o::Opt
    ls::LinSolver
    maxtime::Float64
    function NLGlobalSolver(ls::LinSolver{T},maxtime::Float64)
        lopt = Opt(:LN_SBPLX,ls.ls.params.βlen)
        gopt = Opt(:GN_MLSL_LDS,ls.ls.params.βlen)
        lower_bounds!(lopt,ls.ls.params.βlb)
        upper_bounds!(lopt,ls.ls.params.βub)
        lower_bounds!(gopt,ls.ls.params.βlb)
        upper_bounds!(gopt,ls.ls.params.βub)
        maxtime!(gopt,maxtime)
        xtol_rel!(lopt,1e-5)
        function obj(β::Array{T,1},g=zeros(T,0))
            ls(β)
            ls.ls.calls += 1
            return ls.ls.rss::T
        end
        min_objective!(lopt,obj)
        min_objective!(gopt,obj)
        local_optimizer!(gopt,lopt)
        return new(gopt,ls,maxtime)
    end
end

function call{T<:Base.LinAlg.BlasFloat}(nl::NLGlobalSolver{T},β0::Array{T,1})
    nl.ls.ls.calls = 0
    updateBasis(nl.ls.ls,β0)
    optimize!(nl.o::Opt,nl.ls.ls.β::Array{T,1})
    nl.ls(nl.ls.ls.β)
    return nothing
end

type NLLocalSolver{T<:Base.LinAlg.BlasFloat} <: NLSolver{T}
    o::Opt
    ls::LinSolver{T}
    xtol::Float64
    function NLLocalSolver(ls::LinSolver{T},xtol)
        lopt = Opt(:LN_BOBYQA,ls.ls.params.βlen)
        lower_bounds!(lopt,ls.ls.params.βlb)
        upper_bounds!(lopt,ls.ls.params.βub)
        xtol_rel!(lopt,xtol)
        function obj(β::Array{T,1},g=zeros(T,0))
            ls(β)
            return ls.ls.rss::T
        end
        min_objective!(lopt,obj)
        return new(lopt,ls,xtol)
    end
end

function call{T<:Base.LinAlg.BlasFloat}(nl::NLLocalSolver{T},β0::Array{T,1})
    nl.ls.ls.calls = 0
    updateBasis(nl.ls.ls,β0)
    optimize!(nl.o::Opt,nl.ls.ls.β::Array{T,1})
    nl.ls(nl.ls.ls.β)
    return nothing
end
## Here are some types associated with statistical error estimation

type Stats{T<:Base.LinAlg.BlasFloat}
    N::Int
    ys::Array{T,2}
    βs::Array{T,2}
    αs::Array{T,2}
    yhat::Array{T,1}
    samples::Array{Int,1}
    residual::Array{T,1}
    dist::Distributions.DiscreteUniform
    nl::NLSolver{T}
    function Stats(N::Int, nl::NLSolver{T})
        ys = zeros(T,(nl.ls.ls.params.xlen,N))
        βs = zeros(T,(length(nl.ls.ls.β),N))
        αs = zeros(T,(length(nl.ls.ls.α),N))
        yhat = nl.ls.ls.m*nl.ls.ls.α
        samples = zeros(Int,zeros(yhat))
        residual = copy(nl.ls.ls.residual)
        dist = DiscreteUniform(1,nl.ls.ls.params.xlen)
        return new(ys,βs,αs,yhat,residual,nl,N)
    end
end

call{T<:Base.LinAlg.BlasFloat}(s::Stats{T}) = residualBootStrap!(s)

function resample!{T<:Base.LinAlg.BlasFloat}(v::AbstractVector{T},source::Vector{T},p::Vector{Int})
    L = length(v)
    t = zero(T)
    @inbounds for k=1:L
        t = source[p[k]]
        v[k] = t
    end
    v
end

function genSynthData!{T<:Base.LinAlg.BlasFloat}(s::Stats{T})
    (J,K) = size(s.ys)
    for k=1:K
        rand!(s.dist,s.samples)
        sub_ys = sub(s.ys,:,k)
        resample!(sub_ys,residual,rand(dist,J))
        Base.LinAlg.BLAS.axpy(one(T),s.yhat,sub_ys)
    end
    return nothing
end

function residualBootStrap!(s::Stats)
    (J,K) = size(s.ys)
    genSynthData!(s)
    β0 = copy(s.nl.ls.ls.β)
    for k=1:K
        sub_ys = sub(s.ys,:,k)
        sub_αs = sub(s.αs,:,k)
        sub_βs = sub(s.βs,:,k)
        copy!(s.nl.ls.ls.y,sub_ys)
        s.nl(β0)
        copy!(sub_αs,s.nl.ls.ls.α)
        copy!(sub_βs,s.nl.ls.ls.β)
    end
    return nothing
end




## Utility Functions: Anything that isn't a type, call overload, or lineshape function goes here

function mapLines!{T}(M::Array{T,2},x::Array{T,1},β::Array{T,1},f::Array{BoundedLineShape{T},1})
    (J,K) = size(M)
    @assert length(x)==J
    @assert length(f)==K
    start = 1
    finish = 0
    p = pointer(M)
    ind = 1
    for k=1:K
        start = finish+1
        finish = start+(length(f[k].ub)-1)
        βsub = β[start:finish]
        f[k].f(p,ind,x,βsub)
        ind += J
    end
end

function linObjective{T<:Base.LinAlg.BlasFloat}(α::Array{T,1},ls::LinSystem{T})
    Base.LinAlg.BLAS.blascopy!(length(ls.y),ls.y,stride(ls.y,1),ls.residual,stride(ls.residual,1))
    Base.LinAlg.BLAS.gemv!('N',one(T),ls.m,α,-one(T),ls.residual)
    ls.rss = Base.LinAlg.BLAS.dot(ls.params.xlen,ls.residual,1,ls.residual,1)
    return ls.rss
end

function updateBasis{T<:Base.LinAlg.BlasFloat}(ls::LinSystem{T}, β::Array{T,1})
    Base.LinAlg.BLAS.blascopy!(ls.params.βlen,β,stride(β,1),ls.β,stride(ls.β,1))
    mapLines!(ls.m,ls.params.axis,ls.β,ls.params.f)
    orthogonalize!(ls.m,ls.t,5*eps(T))
end

function initα{T}(ls::LinSystem{T})
    L = length(ls.α)
    @inbounds for k=1:L
        ls.α[k] = one(T)
    end
end

function rangefill!{T<:Integer}(v::Vector{T})
    c = zero(T)
    @inbounds for k=1:length(v)
        c += 1
        v[k] = c
    end
    v
end

vecrange(v::Vector) = maximum(v)-minimum(v)

function colnorm{T<:Base.LinAlg.BlasFloat}(A::AbstractArray{T,2})
    K = size(A,2)
    @assert K>0
    J = size(A,1)
    cnorm = zeros(T,K)
    c = sub(A,:,1)
    cnorm[1] = norm(c)
    for k=2:K
        c = sub(A,:,k)
        cnorm[k] = norm(c)
    end
    return cnorm
end

function rcgsu!{T<:Base.LinAlg.BlasFloat}(Q::Array{T,2},ind::Int,a::AbstractArray{T,1},r::Array{T,1},ϵ::T)
    #stands for: re-normalizing cannonical gram schmidt update
    #this function will overwrite a and r, returning
    K = size(Q,2)
    @assert K>0
    J = size(Q,1)
    @assert ind<=K
    @assert length(r) == J
    Qs = sub(Q,:,1:ind)
    cnorm = colnorm(Qs)
    @assert all(cnorm .< (1.0+ϵ)) #if not true this thing will explode
    # we expect length(p)<<J for this application.
    # I also expect to call rcgsu for changing lengths of p.  
    # So we allocate it inside
    p = zeros(T,ind)
    
    #do one iteration first, then test if more are needed.
    #first normalize the vector to be orthogonalized
    scale!(a,1/norm(a))
    #p stores the inner products
    Base.LinAlg.BLAS.gemv!('T',one(T),Qs,a,zero(T),p)
    #r stores the projection of a onto Q
    Base.LinAlg.BLAS.gemv!('N',one(T),Qs,p,zero(T),r)
    #updates a, removing r from a
    Base.LinAlg.BLAS.axpy!(-one(T),r,a)
    o = maximum(abs(p))
    while o>ϵ
        #we need to iterate a few times over this process in order
        #to achieve a desired level of orthogonality
        #usually 2-3 iterations is enough to hit machine precsion
        Base.LinAlg.BLAS.gemv!('T',one(T),Qs,a,zero(T),p)
        Base.LinAlg.BLAS.gemv!('N',one(T),Qs,p,zero(T),r)
        Base.LinAlg.BLAS.axpy!(-one(T),r,a)
        o = maximum(abs(p))
    end
    scale!(a,1/norm(a))
    return a
end

function orthogonalize!{T<:Base.LinAlg.BlasFloat}(Q::Array{T,2},r::Array{T,1},ϵ::T)
    K = size(Q,2)
    @assert K>0
    J = size(Q,1)
    view = sub(Q,:,1)
    if norm(view) > (1+ϵ)
        scale!(view,1/norm(view)) #inplace scaling
    end
    @assert length(r)==J
    for k=2:K
        view = sub(Q,:,k)
        rcgsu!(Q,k-1,view,r,ϵ)
    end
end

function absdiag!{T}(d::Vector{T},M::Array{T,2})
    K = length(d)
    @assert size(M,2) >= K
    L = size(M,1)
    @inbounds for k=1:K
        d[k] = abs(M[(k-1)*L+k])
    end
end

function numSigElems{T<:AbstractFloat}(d::Vector{T})
    c = zero(Int64)
    t = zero(Bool)
    L = length(d)
    ϵ = L*eps(T)
    @inbounds for k=1:L
        t = d[k]>ϵ
        c = (t) ? c+1 : c
    end
    return c
end

function eye!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2})
    L = minimum(size(A))
    fill!(A,zero(T))
    @inbounds for k=1:L
        A[k,k] = one(T)
    end
    return A
end

function qr!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2},q::Array{T,2},r::Array{T,2},p::Array{Int,1})
    (M,N) = size(A)
    @assert size(q) == (M,N)
    @assert size(r) == (N,N)
    F = qrfact!(A,Val{true}) #pivoted QR
    sf = sub(F.factors,1:N,1:N)
    copy!(r,sf)
    triu!(r)
    Base.LinAlg.LAPACK.ormqr!('L','N',F.factors,F.τ,eye!(q)) #unpack Q into our pre-allocated q
    copy!(p,F.jpvt)
    return nothing
end

function swapcols!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2},swap_vec::Vector{T},source_ind::Int,dest_ind::Int)
    if source_ind == dest_ind
        return
    end
    (M,N) = size(A)
    @assert length(swap_vec)==M
    @assert source_ind<=N && source_ind>0
    @assert dest_ind<=N && dest_ind>0
    sub_source = sub(A,:,source_ind)
    sub_dest = sub(A,:,dest_ind)
    copy!(swap_vec,sub_source)
    copy!(sub_source,sub_dest)
    copy!(sub_dest,swap_vec)
    return A
end

function swaprows!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2},swap_row::Array{T,2},source_ind::Int,dest_ind::Int)
    if source_ind == dest_ind
        return
    end
    (M,N) = size(A)
    @assert size(swap_row,2)==N
    @assert size(swap_row,1)==1
    @assert source_ind<=M && source_ind>0
    @assert dest_ind<=M && dest_ind>0
    sub_source = sub(A,source_ind,:)
    sub_dest = sub(A,dest_ind,:)
    copy!(swap_row,sub_source)
    copy!(sub_source,sub_dest)
    copy!(sub_dest,swap_row)
    return A
end

function permutecols!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2},swap_vec::Vector{T},p::Vector{Int})
    (M,N) = size(A)
    L = length(p)
    @assert length(swap_vec)==M
    @assert maximum(p)<=N
    @assert minimum(p)>0
    @assert L<=N
    t = zero(Int)
    for k=1:L
        source_ind = findfirst(p,k)
        swapcols!(A,swap_vec,source_ind,k)
        t = p[k]
        p[k] = k
        p[source_ind] = t
    end
    A
end

function permuterows!{T<:Base.LinAlg.BlasFloat}(A::Array{T,2},swap_row::Array{T,2},p::Vector{Int})
    (M,N) = size(A)
    L = length(p)
    @assert size(swap_row,2)==N
    @assert size(swap_row,1)==1
    @assert maximum(p)<=M
    @assert minimum(p)>0
    @assert L<=M
    t = zero(Int)
    for k=1:L
        source_ind = findfirst(p,k)
        swaprows!(A,swap_row,source_ind,k)
        t = p[k]
        p[k] = k
        p[source_ind] = t
    end
    A
end






## LineShape Functions

function constOffset!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==0
    t1 = one(T)
    @inbounds for k=1:L
        unsafe_store!(r,t1,istart+k-1)
    end
    return nothing
end

function linearOffset!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==0
    t1 = one(T)
    @inbounds for k=1:L
        unsafe_store!(r,x[k],istart+k-1)
    end
    return nothing
end

function fixedLine!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(fixed)==L
    @inbounds for k=1:L
        unsafe_store!(r,fixed[k],istart+k-1)
    end
    return nothing
end

function lorentzFun!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==2
    t1 = zero(T)
    t2 = β[2]/(2*pi)
    t3 = (β[2]^2)/4
    t4 = zero(T)
    @inbounds for k=1:L
        t1 = (x[k]-β[1])^2
        t4 = t2*(1/(t1+t3))::T
        unsafe_store!(r,t4,istart+k-1)
    end
    return nothing
end

function gaussianFun!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==2
    t1 = zero(T)
    t2 = 1/(β[2]*sqrt(2*pi))
    t3 = 2*(β[2]^2)
    t4 = zero(T)
    @inbounds for k=1:L
        t1 = (x[k]-β[1])^2
        t4 = t2*exp(-t1/t3)::T
        unsafe_store!(r,t4,istart+k-1)
    end
    return nothing
end


function voigtFun!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==3
    t1 = zero(Complex{T})
    t2 = zero(T)
    t3 = (β[3]*sqrt(2*pi))
    @inbounds for k=1:L
        t1 = erfcx(-im * (((x[k]-β[1])+im*β[2])/(β[3]*sqrt(2))))
        t2 = (real(t1)/t3)::T
        unsafe_store!(r,t2,istart+k-1)
    end
    return nothing
end

function gFactor{T}(x::T,x0::T,g0::T,A::T)
    return (2*g0)/(1 + exp(A*(x-x0)))::T
end

function asymmetricVoigtFun!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==4
    t1 = zero(Complex{T})
    t2 = zero(T)
    t3 = zero(T)
    γ = zero(T)
    σ = zero(T)
    @inbounds for k=1:L
        γ = gFactor(x[k],β[1],β[2],β[4])
        σ = gFactor(x[k],β[1],β[3],β[4])
        t1 = erfcx(-im * (((x[k]-β[1])+im*γ)/(σ*sqrt(2))))
        t3 = (σ*sqrt(2*pi))
        t2 = (real(t1)/t3)::T
        unsafe_store!(r,t2,istart+k-1)
    end
    return nothing
end

function asymmetricVoigtFun2!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==4
    t1 = zero(Complex{T})
    t2 = zero(T)
    t3 = (β[3]*sqrt(2*pi))
    γ = zero(T)
    σ = zero(T)
    @inbounds for k=1:L
        γ = gFactor(x[k],β[1],β[2],β[4])
        t1 = erfcx(-im * (((x[k]-β[1])+im*γ)/(β[3]*sqrt(2))))
        t2 = (real(t1)/t3)::T
        unsafe_store!(r,t2,istart+k-1)
    end
    return nothing
end

function asymmetricGaussian!{T}(r::Ptr{T},istart::Int,x::Array{T,1},β::Array{T,1},fixed::Array{T,1})
    L = length(x)
    @assert length(β)==3
    t1 = zero(T)
    t2 = zero(T)
    t3 = zero(T)
    t4 = zero(T)
    σ = zero(T)
    @inbounds for k=1:L
        σ = gFactor(x[k],β[1],β[2],β[3])
        t1 = (x[k]-β[1])^2
        t2 = 2*(σ^2)
        t3 = 1/(σ*sqrt(2*pi))
        t4 = t3*exp(-t1/t2)::T
        unsafe_store!(r,t4,istart+k-1)
    end
    return nothing
end

end
