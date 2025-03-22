function [A,B,C,D,E] = sys_matrix(N)
    % mass-spring system
    M = 2*N;
    T = zeros(M/2,M/2);
    for i = 1:M/2
        T(i,i) = -2;
        if i+1<=M/2
            T(i,i+1)=1;
            T(i+1,i)=1;
        end
    end
    A = [zeros(M/2,M/2), eye(M/2); T, zeros(M/2,M/2)];
    B = [zeros(M/2,M/2);
         eye(M/2)];
    E = [zeros(M/2,M/2);
         eye(M/2)];
    C = [eye(M);
         zeros(M/2,M)];
    D = [zeros(M,M/2)
         eye(M/2)];
end

