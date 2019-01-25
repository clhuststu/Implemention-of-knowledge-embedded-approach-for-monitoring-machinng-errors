clc;
% close all;

total_samples_dim = size(F1,1);                     % the amount of training samples
total_weights_dim = size(F1,2);                     % the amount of dictionary columns

random_index = randperm(total_samples_dim);         % random shuffle the training samples
F1 = F1(random_index,:);
y = y(random_index);


sigma = 0.5;                                        % initilize the variance of the measurement uncertainty
lambda0 =  0.01*sigma*norm(F1'*y,inf);              
gamma = ones(total_weights_dim,1);                  % initilize the hyperparameters
invA0 = sparse(diag(1./gamma));
Max_iter = 600;                                     % maximun iterations
                                      
flag = 0;                                           % iteration termination
i = 0;  
while i<Max_iter
    % E²½
    Abarinv = woodbury(F1,invA0*lambda0);
    Dw = lambda0*Abarinv;                           % the covariance matrix in Eq.(19)
    Ew =  1/lambda0*Dw*F1'*y;                       % the expection of weights in Eq.(19)

    % M²½
    error = y-F1*Ew;
    lambda0 = (norm(error,2)^2+lambda0*(total_weights_dim...     % update the variance of the measurement uncertainty in Eq.(22)
        -sum(1./gamma.*diag(Dw))))/total_samples_dim;
    gamma = Ew.^2+(diag(Dw))+1e-120;                             % update the hyperparameters in Eq.(22)
    invA0 = sparse(diag(1./gamma));
    
    i = i+1;
    weight_est = Ew;
    index = gamma(1:end)./max(abs(gamma))<1e-6;    % remove the terms that approach zero
    weight_est(index) = 0;
    
    report_w(i) = norm(Ew,1);
    report_lambda(i) = sqrt(lambda0)/5.3735*1000;

    if i>1                                        % terminate the iteration when the values of weights converge.
        if abs(report_w(i)-report_w(i-1))/abs(report_w(i))<1e-4...
            && abs(report_lambda(i)-report_lambda(i-1))/abs(report_lambda(i))<1e-4
            flag = flag+1;
        end
    end
    if flag==5
        break;
    end
end
weight_est(abs(weight_est)<1e-3)=0;

% plot Figure 4(a)
figure(1);
set(gcf,'position',[100,200,720,600]);
hold on;
[AX,h1,h2] = plotyy(1:length(report_lambda),report_lambda,...
    1:length(report_lambda),report_w);
h1.LineWidth = 2;
h2.LineWidth = 2;
legend({'\lambda^{1/2}','||{\bf{\mu}}_{\bf{w}}||_1'},'Fontsize',14,'Fontweight','bold');
xlabel('Iteration number','Fontsize',16,'Fontweight','bold');
ylabel(AX(1),'Standard deviation \lambda^{1/2} of prediction noise ({\mu}m)',...
    'Fontsize',16,'Fontweight','bold');
ylabel(AX(2),'{\bf{\it{l}}}_1-norm of the weight vector','Fontsize',16,'Fontweight','bold');
set(AX(1),'Fontsize',16,'Fontweight','bold');
set(AX(2),'Fontsize',16,'Fontweight','bold');
alldatacursors = findall(gcf,'type','hggroup');
set(alldatacursors,'FontSize',16)
box on;
hold off;

% plot Figure 4(b)
figure(2);
set(gcf,'position',[100,200,1440,600]);
hold on;
stem(weight_est);
xlabel('Number of weight vector component','Fontsize',16,'Fontweight','bold');
ylabel('Values of weight vector component','Fontsize',16,'Fontweight','bold');
set(gca,'Fontsize',16,'Fontweight','bold');
box on;
hold off;
