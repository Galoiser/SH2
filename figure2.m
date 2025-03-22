% Figure 2
%%%%%%%%%%%%%%%%%%%%%%%%% Subgradient descent
clear
clc

global A B C D E gamma rho eta W N
N = 2;
A = diag([-1,-2,-3,-4]);
B = [-1 1 0 0 0 0
      0 -1 -1 1 0 0
      1 0 1 -1 -1 1
      0 0 0 0 1 -1];
C = [eye(4);zeros(6,4)];
D = [zeros(4,6);eye(6,6)];
E = eye(4);

gamma = 0.95; % H2 requirement 

% Initialize as LQR gain
Q = C'*C;
R = D'*D;
[Kr, X] = lqr(A,B,1*Q,1*R); % default A-BK
Kr = round(Kr,5);
spr_0 = size(nonzeros(Kr),1);
Kr = -Kr; % A+BK in paper

% for test = 1:2
% K0 = Kr + 0.2*rand(size(Kr));
% while max(real(eig(A+B*K0)))>-1e-4 || gamma^2-J(K0)<=0
%     K0 = Kr + 0.2*rand(size(Kr));
% end

% Initialize main parameters
W = ones(size(Kr)); % weighting matrix
h = 0; % iterator
eta = 0.2; % log barrier penalty

slist1 = [];
slist2 = [];
slist3 = [];

K0 = Kr;

while h < 10
    K = K0;
    slist1 = [slist1 norm(K,1)];
    slist2 = [slist2 J(K)];
    slist3 = [slist3 nnz(round(K,4))]
    

    % Subgradient Descent
    s = 10;
    t = 0.9;
    itr = 0;
    
    while 1==1
        itr = itr+1
        grad = sign(K)+grad_H(K)/(eta*(gamma^2 - J(K)));
        while  logbr(K-s*grad,eta)>logbr(K,eta) || logbr(K-s*grad,eta)>logbr(K-s*t*grad,eta)
            s = s*t; % stepsizes
        end
        if s<1e-12
            break
        end
        K=K-s*grad;
    end
    K0 = K;
    eta = 2*eta;
    h = h + 1
end





%%%%%%%%%%%%%%%%%%%%%%%% Compare to ADMM
global A B C D E gamma rho eta W N
N = 2;
A = diag([-1,-2,-3,-4]);
B = [-1 1 0 0 0 0
      0 -1 -1 1 0 0
      1 0 1 -1 -1 1
      0 0 0 0 1 -1];
C = [eye(4);zeros(6,4)];
D = [zeros(4,6);eye(6,6)];
E = eye(4);

gamma = 0.95; % H2 requirement 




% Initialize as LQR gain
Q = C'*C;
R = D'*D;
[Kr, X] = lqr(A,B,1*Q,1*R); % default A-BK
Kr = round(Kr,5);
spr_0 = size(nonzeros(Kr),1);
Kr = -Kr; % A+BK in paper

% for test = 1:2
% K0 = Kr + 0.2*rand(size(Kr));
% while max(real(eig(A+B*K0)))>-1e-4 || gamma^2-J(K0)<=0
%     K0 = Kr + 0.2*rand(size(Kr));
% end

% Initialize main parameters
W = ones(size(Kr)); % weighting matrix
h = 0; % iterator
eta = 0.2; % log barrier penalty

list1 = [];
list2 = [];
list3 = [];

K0 = Kr;

while h < 10
% Initialize ADMM parameters
    k = 0; % ADMM iterator
    rho = 10; % ADMM regularization 60
    F = 0*K0; % split variable
    K = K0;
    L = 0*K0; % dual variable
    list1 = [list1 norm(K,1)];
    list2 = [list2 J(K)];
    list3 = [list3 nnz(round(K,4))]
    adst1 = [];
    adst2 = [];
    adst1 = [adst1 norm(F-K,inf)];
     adst2 = [adst2 sqrt(J(K))];
   while norm(F-K,inf)>1e-5 && k<300
       K = argminK(K,F,L);
       F = argminF(K,F,L);
       L = L + rho*(F-K);
       k = k+1;
        rho = rho+1;
        rho = min(rho,100);
% 
        adst1 = [adst1 norm(F-K,inf)];
        adst2 = [adst2 sqrt(J(K))];

   end

   
   K0 = K;
   eta = 2*eta;
   h = h + 1
end
% 
   figure(1)

   yyaxis left
   ax = gca;
   ax.YColor = '#0072BD';
   plot(slist3,'color','#0072BD','linewidth',2.0,'LineStyle','--')
   hold on
   plot(list3,'color','#0072BD','linewidth',2.0,'LineStyle','-')
   ylabel('${\|K\|_0}$','Interpreter','latex','fontsize',20)
   ylim([3.5 24.5])

   yyaxis right
   ax = gca;
   ax.YColor = [0.1702    0.3674    0.1203];
   plot(sqrt(slist1),'color',[0.1702    0.3674    0.1203],'linewidth',2.0,'LineStyle','--')
   hold on
   plot(sqrt(list1),'color',[0.1702    0.3674    0.1203],'linewidth',2.0,'LineStyle','-')
   ylabel('${\|K\|_1}$','Interpreter','latex','fontsize',20)
   xlabel('{$h$}','Interpreter','latex','fontsize',22)
  
   hold off
   
%%%%%%%%%%% Functions

function grad = grad_H(K)
    global A B C D E 
    AK = A+B*K;
    CK = C+D*K;
    X = lyap(AK, E*E');
    Y = lyap(AK', CK'*CK);
    grad = 2*(D'*D*K+B'*Y)*X;
end

function sh = J(K)
    global A B C D E 
    AK = A+B*K;
    CK = C+D*K;
    X = lyap(AK, E*E');
    sh = trace(CK*X*CK');
end

function JL = Lagr(K,F,L)
    global A B gamma rho eta W
    if max(real(eig(A+B*K)))<-1e-4 && gamma^2-J(K)>0
        JL = norm(F.*W,1)-log(gamma^2-J(K))/eta...
            +trace(L'*(F-K))+(rho/2)*norm(F-K,2)^2;
    else
        JL = inf;
    end
end

function grad = grad_L(K,F,L)
    global gamma rho eta 
    grad = rho*(K-F)+ grad_H(K)/(eta*(gamma^2 - J(K)))-L;
end

function newK = argminK(K,F,L)
    s = 10;
    t = 0.5;
    while Lagr(K-s*grad_L(K,F,L),F,L)>Lagr(K,F,L) || Lagr(K-s*grad_L(K,F,L),F,L)>Lagr(K-s*t*grad_L(K,F,L),F,L)
        s = s*t; % stepsizes
    end
    newK=K-s*grad_L(K,F,L);
end

function z = sft(b,a)
    if a>b
        z = a-b;
    end
    if abs(a)<=b
        z = 0;
    end
    if a<-b
        z = a+b;
    end
end

function newF = argminF(K,F,L)
    global N W rho
    newF = 0*F;
    for i=1:N
        for j=1:2*N
            newF(i,j) = sft(W(i,j)/rho, K(i,j)-L(i,j)/rho);
        end
    end
end

function ns = logbr(K,eta) 
    global A B gamma 
    if max(real(eig(A+B*K)))<-1e-4 && gamma^2-J(K)>0
        ns = norm(K,1)-log(gamma^2-J(K))/eta;
    else
        ns = 1e8;
    end
end
