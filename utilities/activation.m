function [f_act]=activation(x,act)
% if strcmp(act,'tanh')
%     f_act=tanh(x);
% elseif strcmp(act,'relu')
%     f_act=zeros(size(x));
%     f_act(find(sign(x)>=0))=x(find(sign(x)>=0));
% elseif strcmp(act,'linear')
%     f_act=x; 
% elseif strcmp(act,'sigmoid')
%     f_act = 1 ./ (1 + exp(-x));
% end

if strcmp(act,'t')
    f_act=tanh(x);
%     disp('t')
elseif strcmp(act,'r')
%     f_act=zeros(size(x));
%     f_act(find(sign(x)>=0))=x(find(sign(x)>=0));
    f_act=max(x,0);
%     disp('r')
elseif strcmp(act,'l')
    f_act=x; 
%     disp('l')
elseif strcmp(act,'s')
    f_act = 1 ./ (1 + exp(-x));
%     disp('s')
end