function [ X, L ] = LTLt_unb( variant, X, L )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  variant - algorithmic variant to use
%         X       - skew-symmetric matrix
%         L       - passed in as identity matrix 
%
% Output: X       - Tridiagonal skew-symmetric matrix T
%         L       - Unit lower triangular matrix  
%
% Unblocked algorithms
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

for i=1:n-1

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; R3 = [i+2:n];
    R01 = [1:i]; R23= [i+1:n];

    switch variant
    case 'unb_right'
        % l32 = x31 / chi21
        L( R3, r2 ) = X( R3, r1 ) / X( r2, r1 );

        % x31 = 0
        X( R3, r1) = zeros( size(R3,1), 1 );
    
        % X33 = X33 + ( l32 * x32' - x32 * l32' ) 
        % updating only (strictly) lower triangular part
        X( R3, R3 ) = X( R3, R3 ) + ...
            tril ( L( R3, r2 ) * X( R3, r2 )' - ...
                   X( R3, r2 ) * L( R3, r2 )', -1 ); 
    case 'unb_left'
        % / chi21 \ -:= / l20' lambda21 \ / X00 -x10' \ / l10 \
        % \  x31  /     \ L30     l31   / \ x10   0   / \  1  /
        X( R23, r1 ) = X( R23,r1 ) - ...
            L( R23, R01 ) * ( SkewSym_L( X( R01, R01 ) ) * L( r1, R01 )' );

        % l32 := x31 / chi21
        L( R3, r2 ) = X( R3, r1 ) / X( r2, r1 );

        % x31 := 0
        X( R3, r1 ) = zeros( size( R3,1 ), 1 );
    otherwise
        fprintf( "urecognized version %s\n", version);
    end
end

end
