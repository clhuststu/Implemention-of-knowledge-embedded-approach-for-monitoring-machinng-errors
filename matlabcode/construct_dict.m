function A = construct_dict(X,Y,basisWidth)
    m = size(X,1);
    n = size(Y,1);

    D2 = (sum(((X.^2)), 2) * ones(1,n)) + (ones(m, 1) * sum(((Y.^2)),2)') - ...
        2*X*Y';
    A = [exp(-D2/basisWidth) ones(m,1)];
end