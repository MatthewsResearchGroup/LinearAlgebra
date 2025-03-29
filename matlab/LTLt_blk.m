function [ X, L ] = LTLt_blk( variant, X, L, nb )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  variant - algorithmic variant to use
%         X       - skew-symmetric matrix
%         L       - passed in as identity matrix. 
%         nb      -   algorithmic block size
%
% Output: X       - Tridiagonal skew-symmetric matrix T
%         L       - Unit lower triangular matrix.  
%
% Blocked algorithms
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

if m <= 2 
    return
end

switch variant
case 'blk_fused_rightb'
    L(3:m,2) = X( 3:m, 1) / X( 2,1 );
    X( 3:m, 1 ) = zeros( m-2, 1 );
end

for i=1:nb:m-1
    % Determine next block size
    ib = min( m-i, nb );
 
    % Set the various ranges
    R01 = [1:i];
    R12 = [i:i+ib-1]; R123 = [i:i+ib]; R1234 = [i:m]; 
    R23 = [i+1:i+ib]; R234 = [i+1:m]; 
    r3 = [i+ib]; R34 = [i+ib:m];  
    R4 = [i+ib+1:m];
    R23p = [i+1:min(i+ib+1,m-1)];    % R23 plus one extra column
     
    switch variant
    case 'blk_right'
        [ X( R1234,R1234 ), L( R1234,R1234 ) ] = ...
                LTLt_unb_left_trunc( true, X( R1234,R1234 ), ...
                L( R1234,R1234 ), ib );

        X( R34, R34 ) = X( R34, R34 ) - tril( L( R34,R23 )  * ...
                   SkewSym_L( X( R23,R23) ) * L( R34,R23 )', -1 );

        X( R4,R4 ) = X(R4,R4 ) + tril( L( R4,r3 ) * X( R4,r3 )'...
                                     - X( R4,r3 ) * L( R4,r3 )', -1 );
    case 'blk_fused_righta'  
        [ X( R1234,R1234 ), L( R1234,R1234 ) ] = ...
            LTLt_unb_left_trunc( false, X( R1234,R1234 ), ...
                                        L( R1234,R1234 ), ib );

        X( R34, R34 ) = X( R34, R34 ) - tril( L( R34,R123 ) ...
               * SkewSym_L( X( R123,R123) ) * L( R34,R123 )', -1 );
    case 'blk_fused_rightb'
        [ X( R234,R234 ), L( R234,R234 ) ] = ...
            LTLt_unb_left_trunc( false, X( R234,R234 ), L( R234,R234 ), ...
                                 ib );
        
        X( R4,R4 ) = X( R4,R4 ) - tril( L( R4,R23p ) * ...
                 ( SkewSym_L( X( R23p,R23p ) ) * L( R4,R23p )' ) );
    case 'blk_left'
        X( R1234,R12 ) = X( R1234,R12 ) - tril( L( R1234,R01 ) * ...
                  ( SkewSym_L( X( R01,R01 ) ) * L( R12,R01 )' ) );

        [ X( R1234,R1234 ), L( R1234,R1234 ) ] = ...
            LTLt_unb_left_trunc( false, X( R1234,R1234 ), ...
                                 L( R1234,R1234 ), ib );
    otherwise
        fprintf( "urecognized version %s\n", version);
    end

end

end