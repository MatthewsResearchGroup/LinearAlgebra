function [ X, L, W ] = LTLt_unb_Wimmer_trunc( X, L, W, k )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  X - skew-symmetric matrix
%         L - passed in as identity matrix. 
%
% Output: X - Tridiagonal skew-symmetric matrix T
%         L - Unit lower triangular matrix.  
%         W - matrix in which "w vectors" are returned to bloked algorithm
%
% Unblocked Wimmer's algorithm, restricted to only update the first k
% columns of X and return the various "w vectors".  
%
% copyright 2023, 2024 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );
assert( mod( m, 2 ) == 0, 'size must be even' );

W = zeros( size( W ) );
if m<=2 
    return
end

for i=1:2:min(m-2,k)

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; r3 = [i+2:i+2]; R4 = [i+3:n];
    R34 = [i+2:n]; R4trunc = [i+3:min(k+1,m)];

    % / lambda32 \ = / chi31 \ / chi21
    % \    l42   /   \  x41  /
    L( R34,r2 ) = X( R34,r1 ) / X( r2,r1 );

    % / chi31 \ = / 0 \
    % \  x41  / . \ 0 /
    X( R34,r1 ) = zeros( size(R34, 1), 1 );

    % l43 = x42 / chi32
    L( R4,r3 ) = X( R4,r2 ) / X( r3,r2 );

    % x42 = 0
    X( R4,r2 ) = zeros( size(R4, 1), 1 );

    % w43 = x43 - tau32 * lambda32 * l43
    % Store the vector which with the skew-symm rank-2 update is performed
    % for use in the blocked algorithm
    W( R4,r3 ) = X( R4,r3 ) - X( r3,r2 ) * L( r3,r2 ) * L( R4,r3 );

    % x43 = w43 + tau32 * l42 = x43 + tau32 * l42 - tau32 * lambda32 * l43
    X( R4,r3 ) = W( R4,r3 ) + X( r3,r2 ) * L( R4,r2 );

    % X44 = X44 + l43 * w43' - w43 * l43' = 
    %       X44 + ( l43 * ( x43 - tau32 * l42 )' - ...
    %                 ( x43 - tau32 * l42 ) l43' 
    X( R4,R4trunc ) = X( R4, R4trunc ) + ...
        tril ( L( R4,r3 ) * W( R4trunc,r3 )' - W( R4,r3 ) * ...
        L( R4trunc,r3 )', -1 );

end

end
