function [ X, L ] = LTLt_unb_2_step( X, L )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  X - skew-symmetric matrix
%         L - passed in as identity matrix. 
%
% Output: X - Tridiagonal skew-symmetric matrix T
%         L - Unit lower triangular matrix.  
%
% Unblocked 2-step algorithm
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );
assert( mod( m, 2 ) == 0, 'size must be even' );
assert( n >= 2, 'size must be at least 4' );

for i=1:2:n-2

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; r3 = [i+2:i+2]; R4 = [i+3:n];
    R34 = [i+2:n];

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

    X( R4,R4 ) = X( R4, R4 ) + ...
        tril ( L( R4,r3 ) * X( R4,r3 )' - X( R4,r3 ) * L( R4,r3 )', -1 );

    % x43 x43 + tau32 * l42 - tau32 * lambda32 * l43
    X( R4,r3 ) = X( R4,r3 ) - X( r3,r2 ) * L( r3,r2 ) * L( R4,r3 ) + ...
        X( r3,r2 ) * L( R4,r2 );

end

end
