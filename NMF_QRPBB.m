function [W,H,iter,elapse,HIS]=NMF_QRPBB(V,r,varargin)

% Quadratic regularization projected Barzilai--Borwein method for nonnegative matrix factorization
%
% This code solves the following problems: given V and r, find W and H such that
%     minimize 1/2 * || V-WH ||_F^2 subject to W>=0 and H>=0. 
%
% Reference: Yakui Huang, Hongwei Liu, and Shuisheng Zhou: "Quadratic regularization projected Barzilai--Borwein method 
% for nonnegative matrix factorization", Data Mining and Knowledge Discovery, 2015, 29(6): 1665-1684.
%
% Written by Yakui Huang (huangyakui2006@gmail.com)

% This code applies to V (m x n) with m > n. If m <=n, applies it to V^T is faster.


% <Inputs>
%        V : Input data matrix (m x n)
%        r : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value pairs)
%        MAX_ITER : Maximum number of iterations. Default is 1,000.
%        MIN_ITER : Minimum number of iterations. Default is 1.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x r) initial value for W.
%        H_INIT : (r x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-7. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging purpose) - History of computation is returned by 'HIS' variable.
%                  2 (debugging purpose) - History of computation is additionally printed on screen.
% <Outputs>
%        W : Obtained basis matrix (m x r)
%        H : Obtained coefficients matrix (r x n)
%        iter : Number of iterations
%        elapse : CPU time in seconds
%        HIS : (debugging purpose) History of computation
%
% <Usage Examples>
%        >>V=rand(100);
%        >>NMF_QRPBB(V,10)
%        >>NMF_QRPBB(V,20,'verbose',1)
%        >>NMF_QRPBB(V,30,'verbose',2,'w_init',rand(m,r))
%        >>NMF_QRPBB(V,5,'verbose',2,'tol',1e-5)

if ~exist('V','var'),    error('please input the sample matrix.\n');    end
if ~exist('r','var'),    error('please input the low rank.\n'); end

[m,n]=size(V);

% Default setting
MaxIter=1000;
MinIter=10;
MaxTime=100000;
W0=rand(m,r);
H0=rand(r,n);
tol=1e-7;
verbose=0;

% Read optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER',    MaxIter=varargin{i+1};
            case 'MIN_ITER',    MinIter=varargin{i+1};
            case 'MAX_TIME',    MaxTime=varargin{i+1};
            case 'W_INIT',      W0=varargin{i+1};
            case 'H_INIT',      H0=varargin{i+1};
            case 'TOL',         tol=varargin{i+1};
            case 'VERBOSE',     verbose=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

ITER_MAX=1000;      % maximum inner iteration number (Default)
ITER_MIN=1;         % minimum inner iteration number (Default)

% Initialization
W=W0; H=H0;
WtW=W'*W; WtV=W'*V;
HHt=H*H'; HVt=H*V';
GradH=WtW*H-WtV;
GradW=W*HHt-HVt';

init_delta = norm([GradW; GradH'],'fro');
tolH=max(tol,1e-3)*init_delta;
tolW=tolH;               % Stopping tolerance
constV=sum(sum(V.^2));

% Historical information
HIS.niter=0;
HIS.t=0;
HIS.f=sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H));
HIS.p=init_delta;

% Iterative updating
elapse=cputime;
W=W';
for iter=1:MaxIter,   

    % Optimize W with H fixed
    [W,iterW,GradW] = QRPBB(W,HHt,HVt,ITER_MAX,ITER_MIN,tolW);
    if iterW<=ITER_MIN,
        tolW=tolW/10;
    end
    WtW=W*W'; WtV=W*V;
    
    % Optimize H with W fixed
    [H,iterH,GradH] = QRPBB(H,WtW,WtV,ITER_MAX,ITER_MIN,tolH);
    if iterH<=ITER_MIN,
        tolH=tolH/10;
    end
    HHt=H*H';   HVt=H*V';
%     GradW=HHt*W-HVt;

    HIS.niter=HIS.niter+iterH+iterW;
    delta = norm([GradW(GradW<0 | W>0); GradH(GradH<0 | H>0)]);

    % Output running detials
    if verbose,
        HIS.f=[HIS.f,sum(sum(WtW.*HHt))-2*sum(sum(WtV.*H))];
        HIS.t=[HIS.t,cputime-elapse];
        HIS.p=[HIS.p,delta];
        if (verbose==2) && (rem(iter,10)==0),
            fprintf('%d:\tstopping criteria = %e,\tobjective value = %e.\n', iter,delta/init_delta,HIS.f(end)+constV);
        end
    end
    
    % Stopping condition
    if (delta<=tol*init_delta && iter >= MinIter) || HIS.t(end)>=MaxTime,
        break;
    end 
end
W=W';
elapse=cputime-elapse;

if verbose,
    HIS.f=0.5*(HIS.f+constV);
    if verbose==2,
        fprintf('\nFinal Iter = %d,\tFinal Elapse = %f.\n', iter,elapse);
    end
end



function [x,iter,gradx] = QRPBB(x0,WtW,WtV,iterMax,iter_Min,tol)
% Quadratic regularization projected Barzilai--Borwein method for the Nonnegative 
% Least Squares Problem: min 1/2 * \|V-Wx\|_{F}^2 subject to x>=0, which is
% equivalent to min 1/2 * <x,WtWx> - <x,WtV> subject to x>=0.

mm=5;       %mm nonmonotone line search parameter
lamax=10^20; lamin=10^-20;
gamma=10^-4;
rho = 0.25; % step length factor

x = x0;     % Initialization
delta0=-sum(sum(x.*WtV));
dQd0 = sum(sum((WtW*x).*x));
f0=delta0+0.5*dQd0;
fn = f0;
lambda=1;

L = 1/norm(full(WtW));    % Lipschitz constant
gradx = WtW*x - WtV;      % Gradient
for iter=1:iterMax,

    % Stopping criteria
    if iter>=iter_Min,
        pgn=norm(gradx(gradx < 0 | x >0));
        if pgn<=tol,
            break;
        end
    end
 
    % caculate a point by using the Lipschitz constant
    dx = max(x - L.*gradx, 0)-x;
    dgradx = WtW*dx;
    delta = dx(:)'*gradx(:);
    dQd = dx(:)'*dgradx(:);
    x = x + dx;
    fn = fn + delta + 0.5*dQd;
    gradx = gradx + dgradx;
    
    % run a projected Barzilai--Borwein step
    func(iter)=fn;
    jj=min(iter-1,mm-1);
    fmax=max(func(iter-jj:iter));
    
    dx = max(x - lambda.*gradx, 0)-x;
    dgradx = WtW*dx;
    delta = dx(:)'*gradx(:);
    dQd = dx(:)'*dgradx(:);
    fn = func(iter) + delta + 0.5*dQd;
    alpha=1;
    while (fn > fmax + alpha*gamma*delta)
        % Use Backtracking Line Search
        alpha=rho*alpha;
        fn = func(iter)+alpha*delta+0.5*alpha^2*dQd;
    end
    x = x+alpha.*dx;
        
    % Compute the BB steplength 
    sty = dQd;   
    gradx = gradx + alpha.*dgradx;
    if sty > 0
        sts = dx(:)'*dx(:);
        lambda=min(lamax,max(lamin,sts/sty));
    else
        lambda=lamax;
    end   
end

if iter==iterMax,
    fprintf('Max iter in QRPBB\n');
end

