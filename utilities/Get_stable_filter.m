function a_stable = Get_stable_filter(a_unstable)

% Function to get linear prediction (LP) coefficients which form a stable all pole filter 
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      a_unstable     : LP coefficients which form an unstable all pole filter
%
%   Output parameters:
%      a_stable       : LP coefficients of a stable filter
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2018 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p_unstable] = pzmap(tf(1,a_unstable));
p_stable = p_unstable;

poles_real_parts = round(abs(real(p_unstable)),3);

% replace unstable real poles first
loc_real_pole = find(  poles_real_parts >=1 & abs(real(p_unstable))==abs(p_unstable)); 
% a pole is real if it real part = its magnitude
p_stable(loc_real_pole) = 0.98;

r = round(abs(p_stable),3);

loc = find(r>=1);  % find locations of unstable complex poles

j=1;
for i=1:length(loc)/2;
    pole = p_unstable(loc(j));
    polenew = 0.98*exp(sqrt(-1)*angle(pole));
    
    p_stable(loc(j)) = polenew;
    p_stable(loc(j+1)) = conj(polenew);
    
    j=j+2;
end
a_stable = poly(p_stable);
end