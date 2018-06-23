function X1 = memory_inclusion_delta_ext(X, dimX, hlen)

% Convert to a column vector
if size(X,1)==1
    X=X';
end

X1 = reshape(X,dimX,2*hlen+1);
X1_deltas = Deltas(X1,hlen);
X1 = [X1(:,hlen+1);X1_deltas(:,hlen+1)];
