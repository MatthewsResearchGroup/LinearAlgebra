function [ X, L ] = LTLt_unb_right_trunc( X, L, k )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  X - skew-symmetric matrix
%         L - passed in as identity matrix. 
%
% Output: X - Tridiagonal skew-symmetric matrix T
%         L - Unit lower triangular matrix.  
%
% Unblocked right-looking variant
%
% This version only updates the first k columns, leaving the remainder of
% the matrix untouched
%
% copyright 2023, 2024 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

for i=1:min(n-1,k)

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; R3 = [i+2:n];
    % Create a truncated range so that only first k columns are updated.
    R3trunc = [i+2:min(n,k)];
    
    % l32 = x31 / chi21
    L( R3, r2 ) = X( R3, r1 ) / X( r2, r1 );
    % x31 = 0
    X( R3, r1) = zeros( size(R3,1), 1 );
    
    % X33 = X33 + ( l32 * x32' - x32 * l32' ) 
    % updating only (strictly) lower triangular part and do not touch
    % beyond the first k columns of X and k+1 column of L
    X( R3, R3trunc ) = X( R3, R3trunc ) + ...
        tril ( L( R3, r2 ) * X( R3trunc, r2 )' - ...
               X( R3, r2 ) * L( R3trunc, r2 )', -1 );
   
end

end
