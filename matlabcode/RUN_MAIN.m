%% initialization
clc;
format short;

%% Example
example = 'Singleblade';
method = {'KEM','GKM'};
BasisWidth = 1;
%% 
switch example
case 'Singleblade'
    parameter.method = method{1}; 
    %% start of reading data from file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the train set
    load('trainset.mat');
    ym = trainset.ym;               % machining errors
    Fx = trainset.Fx;               % cutting forces
    Fy = trainset.Fy;
    Fz = trainset.Fz;
    u = trainset.u;                 % locations of measured points
    v = trainset.v;
    fz = trainset.fz;               % cutting parameters
    ap = trainset.ap;
    alpha = trainset.alpha;         % tool rake angles
    beta = trainset.beta;
    
    load('norm_work.mat');          % The direction cosines of the surface normal at these points
    norm_work = kron(ones(84,1),norm_work);
    np_x = norm_work(:,1);
    np_y = norm_work(:,2);
    np_z = norm_work(:,3);
    
    tmp1 = tan(alpha/180*pi);       % The direction cosines of the tool orientation                      
    tmp2 = tan(-beta/180*pi);
    nt_y = -sqrt(1./(tmp1.^2+1+tmp2.^2));
    nt_x = -tmp1.*nt_y;
    nt_z = tmp2.*nt_y;

    switch parameter.method
        case 'GKM'
            features = [Fx, Fy, Fz, u, v, alpha, beta, fz, ap];           % construct the feature vector   
            [features,PS_features] = mapminmax(features',0,1);            % normalize the values of each column into the range (0, 1)
            features = features';           
            kernel_features = features;
            
            F1 = construct_dict(features,kernel_features,BasisWidth);            % construct the disctionary matrix
            [F1,PS_F1] = mapminmax(F1',0,1);                              % normalize the columns of the dictionary matrix 
            [y, PS_me] = mapminmax(ym',0,1);                              % normalize the values of the measured machining errors 
            F1 = F1';
            y = y';

        case 'KEM'
            W = Shapefunction(u,v);                                       % The expansion basis functions of flexibility distribution
            [m,n] = size(W);
            
            F1 = [W.*repmat(np_y,1,n), W.*repmat(np_x,1,n),...            % construct the disctionary matrix
                W.*repmat(Fy,1,n).*repmat(np_y,1,n), W.*repmat(Fx,1,n).*repmat(np_x,1,n),...
                Fx.*np_x+Fy.*np_y+Fz.*np_z-(Fx.*nt_x+Fy.*nt_y+Fz.*nt_z).*(nt_x.*np_x+...
                nt_y.*np_y+nt_z.*np_z), alpha, beta, fz, ap, ones(m,1)];
            
            [F1,PS_F1] = mapminmax(F1',0,1);                              % normalize the columns of the dictionary matrix 
            [y, PS_me] = mapminmax(ym',0,1);                              % normalize the values of the measured machining errors 
            F1 = F1';
            y = y';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the test set
    load('testset.mat');
    ym = testset.ym;               % machining errors
    Fx = testset.Fx;               % cutting forces
    Fy = testset.Fy;
    Fz = testset.Fz;
    u = testset.u;                 % locations of measured points
    v = testset.v;
    fz = testset.fz;               % cutting parameters
    ap = testset.ap;
    alpha = testset.alpha;         % tool rake angles
    beta = testset.beta;
    
    load('norm_work.mat');         % The direction cosines of the surface normal at these points
    norm_work = kron(ones(42,1),norm_work);
    np_x = norm_work(:,1);
    np_y = norm_work(:,2);
    np_z = norm_work(:,3);
    
    tmp1 = tan(alpha/180*pi);      % The direction cosines of the tool orientation                      
    tmp2 = tan(-beta/180*pi);
    nt_y = -sqrt(1./(tmp1.^2+1+tmp2.^2));
    nt_x = -tmp1.*nt_y;
    nt_z = tmp2.*nt_y;

    switch parameter.method
        case 'GKM'
            features = [Fx, Fy, Fz, u, v, alpha, beta, fz, ap];           % construct the feature vector   
            features = mapminmax('apply',features',PS_features)';         % normalize the values of each column into the range (0, 1)
            
            F1_test = construct_dict(features,kernel_features,BasisWidth);       % construct the disctionary matrix
            F1_test = mapminmax('apply',F1_test',PS_F1)';                 % normalize the columns of the dictionary matrix 
            y_test = mapminmax('apply',ym',PS_me)';                       % normalize the values of the measured machining errors 
            run('RUN_SOLVER.m')
        case 'KEM'
            W = Shapefunction(u,v);                                       % The expansion basis functions of flexibility distribution
            [m,n] = size(W);
            
            F1_test = [W.*repmat(np_y,1,n), W.*repmat(np_x,1,n),...       % construct the disctionary matrix
                W.*repmat(Fy,1,n).*repmat(np_y,1,n), W.*repmat(Fx,1,n).*repmat(np_x,1,n),...
                Fx.*np_x+Fy.*np_y+Fz.*np_z-(Fx.*nt_x+Fy.*nt_y+Fz.*nt_z).*(nt_x.*np_x+...
                nt_y.*np_y+nt_z.*np_z), alpha, beta, fz, ap, ones(m,1)];

            F1_test = mapminmax('apply',F1_test',PS_F1)';                 % normalize the columns of the dictionary matrix 
            y_test = mapminmax('apply',ym',PS_me)';                       % normalize the values of the measured machining errors 
            run('RUN_SOLVER.m')
    end
end

