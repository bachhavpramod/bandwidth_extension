function [ res ] = pesq_mex_vec( reference_sig, degraded_sig, Fs )
%PESQ_MEX_VEC Accepts vectors for a mex compiled version of the objective Perceptual Evaluation of Speech Quality measure
% 
% Syntax:	[ res ] = pesq_mex_vec( reference_sig, degraded_sig, Fs )
% 
% Inputs: 
% 	reference_sig - Reference (clean, talker, sender) speech signal
% 	degraded_sig - Degraded (noisy, listener, receiver) speech signal
% 	Fs - Sampling Frequency
% 
% Outputs: 
% 	res - MOS-LQO result for wideband
% 
% See also: pesq2mos.m

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2016
% Date: 16 June 2016
% Revision: 0.2
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Added on 21st Feb 2018, because I found that, if the lengths of 2 signals inp to pesqmain2 are diff, then the resul
% is diff than in case if they are of same length

% Convert to row vectors
if size(reference_sig,2)==1
    reference_sig = reference_sig';
end
if size(degraded_sig,2)==1
    degraded_sig = degraded_sig';
end

m= min(length(reference_sig), length(degraded_sig));
degraded_sig= degraded_sig(1:m);
reference_sig = reference_sig(1:m);

tmpref = 'abcd.wav';
tmpdeg ='defg.wav';

audiowrite( tmpref, reference_sig, Fs);
audiowrite( tmpdeg, degraded_sig, Fs);

res = pesqmain2(['+' num2str(Fs)], ...
                    '+wb', ...
                    tmpref, ...
                    tmpdeg);
                
delete( tmpref, tmpdeg );
end
