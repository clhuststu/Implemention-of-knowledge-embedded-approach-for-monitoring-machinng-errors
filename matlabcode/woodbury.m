 function Abarinv = woodbury(A, P)

%     Abar=A'*A + P; 
%     Abarinv=inv(Abar);

p = 1e-12;

[m,n] = size(A);

if m>n
    AtA = A'*A;
    AtA = AtA+P;
    ep =  p*max(AtA(:)); 
    Abarinv=inv(AtA+ep*eye(n));
else
     Pinv = sparse(inv(P));
     P_tmp = eye(m)+A*Pinv*A';
     ep =  p*max(P_tmp(:)); 
     iP = P_tmp +ep*eye(m);
     tmp2 = A'*(iP\A);
     Abarinv = Pinv - Pinv*tmp2*Pinv;
end
    
