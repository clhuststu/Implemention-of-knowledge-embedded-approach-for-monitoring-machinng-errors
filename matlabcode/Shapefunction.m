function W = Shapefunction(u,v)

m = size(u,1);
Numof = [10 10];
W = zeros(m,(Numof(2)*2-1)*Numof(1));

for i=1:Numof(1)
    % the power basis
    if i==1
        f1 = @(x)(x.^1);
        V = f1(v);
    elseif i==2
        f1 = @(x)(x.^2);
        V = f1(v);
    else
        tmp1 = quadgk(@(x)x.^i.*f1(x),0,1);
        tmp2 = quadgk(@(x)f1(x).*f1(x),0,1);
        f2 = @(x)(x.^i-tmp1/tmp2*f1(x));
        V = f2(v);
        f1 = f2;
    end
    
    % the triangle basis
    for j=1:Numof(2)
        if j==1
            W(:,j+(i-1)*(Numof(2)*2-1)) = V;
        else
            W(:,2*j-2+(i-1)*(Numof(2)*2-1)) = V.*cos((j-1)*pi*(2*u-1));
            W(:,2*j-1+(i-1)*(Numof(2)*2-1)) = V.*sin((j-1)*pi*(2*u-1));
        end
    end
end  
end


