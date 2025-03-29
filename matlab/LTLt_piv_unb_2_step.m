function [ X, L, p ] = LTLt_piv_unb_2_step( X, L )

% LTLt_piv_unb_Wimmer Compute permutation vector p, tridiagonal matrix T 
% and unit lower triangular matrix L such that
%
%         P(p) X P(p)' = L * T * L', where 
%
% Input:  X - skew-symmetric matrix, stored in the strictly lower
%             triangular part.
%         L - Lower triangular matrix L passed in as identity matrix. 
%
% Output: X - Tridiagonal skew-symmetric matrix T, stored in the strictly
%             lower triangular part.  Upper triangular part is left
%             untouched.
%         L - Unit lower triangular matrix.
%         p - pivot vector.
%
% Unblocked Wimmer's algorithm with pivoting
%
% size of X must be even.
%
% copyright 2023, 2024 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );
assert( mod( m, 2 ) == 0, 'size must be even' );
assert( n >= 2, 'size must be at least 4' );

% Create pivot vector
p = zeros( m, 1 );

for i=1:2:m-2

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; r3 = [i+2:i+2]; R4 = [i+3:m];
    R01 = [1:i]; R012 = [1:i+1]; R234 = [i+1:n]; R34 = [i+2:m];

    % Determine which row to pivot to top
    p( r2, 1 ) = Iamax( X( R234, r1 ) );

    % Pivot current column
    X( R234,r1 ) = Apply_P( 'left', p( r2, 1 ), X( R234,r1 ) );

    % Pivot rest of X
    X( R234,R234 ) = SkewSymm_Apply_P( p( r2, 1 ), X( R234, R234 ) );

    % Pivot L
    L( R234, R01 ) = Apply_P( 'left', p( r2, 1 ), L( R234, R01 ) );

    % / lambda32 \ = / chi31 \ / chi21
    % \    l42   /   \  x41  /
    L( R34,r2 ) = X( R34,r1 ) / X( r2,r1 );

    % / chi31 \ = / 0 \
    % \  x41  / . \ 0 /
    X( R34,r1 ) = zeros( size(R34, 1), 1 );

    % Determine which row to pivot to top for next column
    p( r3, 1 ) = Iamax( X( R34, r2 ) );

    % Pivot next column
    X( R34,r2 ) = Apply_P( 'left', p( r3, 1 ), X( R34,r2 ) );

    % Pivot rest of X
    X( R34,R34 ) = SkewSymm_Apply_P( p( r3, 1 ), X( R34, R34 ) );

    % Pivot L
    L( R34, R012 ) = Apply_P( 'left', p( r3, 1 ), L( R34, R012 ) );

    % l43 = x42 / chi32
    L( R4,r3 ) = X( R4,r2 ) / X( r3,r2 );

    % x42 = 0
    X( R4,r2 ) = zeros( size(R4, 1), 1 );

    % w43 = x43 - tau32 * lambda32 * l43
    w43 = X( R4,r3 ) - X( r3,r2 ) * L( r3,r2 ) * L( R4,r3 );

    % x43 = w43 + tau32 * l42 = x43 + tau32 * l42 - tau32 * lambda32 * l43
    X( R4,r3 ) = w43 + X( r3,r2 ) * L( R4,r2 );
   
    % X44 = X44 + l43 * w43' - w43 * l43' = 
    %       X44 + ( l43 * ( x43 - tau32 * l42 )' - ...
    %                 ( x43 - tau32 * l42 ) l43' 
    % updating only (strictly) lower triangular part
    X( R4,R4 ) = X( R4, R4 ) + ...
        tril ( L( R4,r3 ) * w43' - w43 * L( R4,r3 )', -1 );
   
end

end
