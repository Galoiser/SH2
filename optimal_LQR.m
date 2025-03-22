
% N = 20;
% [A,B,C,D,E] = sys_matrix(N);

A = diag([-1,-2,-3,-4]);
B = [-1 1 0 0 0 0
      0 -1 -1 1 0 0
      1 0 1 -1 -1 1
      0 0 0 0 1 -1];
C = [eye(4);zeros(6,4)]
D = [zeros(4,6);eye(6,6)];
E = eye(4);

Q = C'*C;
R = D'*D;
[K, X] = lqr(A,B,1*Q,1*R);
K = round(K,4)

% for i=1:N
%     for j = 1:2*N
%         if K(i,j)~=0
%             K(i,j)=1;
%         end
%     end
% end
% heatmap(abs(K),'CellLabelColor','none','ColorbarVisible','off')%,'GridVisible','off'
% axs = struct(gca); %ignore warning that this should be avoided
% cb = axs.Colorbar;
% cb.FontSize = 12;
% Figure 25:11.5 cm