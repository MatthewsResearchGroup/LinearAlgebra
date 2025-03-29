function [ X, L, p ] = LTLt_piv_blk( variant, X, L, nb )

% LTLt_piv_blk Compute permutation vector p, tridiagonal matrix T,
% and unit lower triangular matrix L such that
%
%         P(p) X P(p)' = L * T * L', where 
%
% Input:  X - skew-symmetric matrix, stored in the strictly lower
%             triangular part.
%         L - Lower triangular matrix L passed in as identity matrix. 
%         nb - algorithmic block size.
%
% Output: X - Tridiagonal skew-symmetric matrix T, stored in the strictly
%             lower triangular part.  Upper triangular part is left
%             untouched.
%         L - Unit lower triangular matrix.
%         p - pivot vector.
%
% Blocked right-looking variants with pivoting
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

% Create pivot vector
p = zeros( m, 1 );

if m <= 2 
    return
end

switch variant
    case 'piv_blk_right2b'
    % Determine which row to pivot to top
    p( 2, 1 ) = Iamax( X(2:m, 1 ) );
       
    % Pivot current column
    X( 2:m,1 ) = Apply_P( 'left', p( 2, 1 ), X( 2:m,1 ) );

    % Pivot rest of X
    X( 2:m,2:m ) = SkewSymm_Apply_P( p( 2, 1 ), X( 2:m,2:m ) );

    L(3:m,2) = X( 3:m, 1) / X( 2,1 );

    X( 3:m, 1 ) = zeros( m-2, 1 );
end

for i=1:nb:m-1
    % Determine next block size
    ib = min( m-i, nb );
 
    % Set the various ranges
    R0 = [1:i-1]; R01 = [1:i]; R012 = [1:i+ib-1];
    R12 = [i:i+ib-1]; R123 = [i:i+ib]; R1234 = [i:m]; 
    R23 = [i+1:i+ib]; R234 = [i+1:m]; 
    r3 = [i+ib]; R34 = [i+ib:m];  
    R4 = [i+ib+1:m];
    R23p  = [i+1:min(i+ib+1,m)];    % R23 plus one extra 
    R2p3p  = [i+2:min(i+ib+1,m-1)];   % R23 less the first plus one extra 
    R2p34 = [i+2:m];                  % R234 less the first

    switch variant
    case 'piv_blk_right'
        [ X( R1234,R1234 ), L( R1234,R1234 ), p( R123, 1 ) ] = ...
            LTLt_piv_unb_left_trunc( true, X( R1234,R1234 ), ...
            L( R1234,R1234 ), p( R123,  1 ), ib );
    
        L( R234, R01 ) = Apply_P( 'left', p( R23, 1 ), L( R234, R01 ) );
        
        X( R34, R34 ) = X( R34, R34 ) - tril( L( R34,R23 )  * ...
                   SkewSym_L( X( R23,R23) ) * L( R34,R23 )', -1 );

        X( R4,R4 ) = X(R4,R4 ) + tril( L( R4,r3 ) * X( R4,r3 )'...
                                     - X( R4,r3 ) * L( R4,r3 )', -1 );

        case 'piv_blk_right2a'
        [ X( R1234,R1234 ), L( R1234,R1234 ), p( R123,  1 ) ] = ...
            LTLt_piv_unb_left_trunc( false, X( R1234,R1234 ), ...
                                    L( R1234,R1234 ), p( R123,  1 ), ib );
        
        L( R234, R0 ) = Apply_P( 'left', p( R23, 1 ), L( R234, R0 ) );   

        X( R34, R34 ) = X( R34, R34 ) - tril( L( R34,R123 ) ...
               * SkewSym_L( X( R123,R123) ) * L( R34,R123 )', -1 );
    
        case 'piv_blk_right2b'
        [ X( R234,R234 ), L( R234,R234 ), p( R23p,  1 ) ] = ...
            LTLt_piv_unb_left_trunc( false, X( R234,R234 ), ...
                                    L( R234,R234 ), p( R23p,  1 ), ib );

        L( R2p34, R01 ) = Apply_P( 'left', p( R2p3p, 1 ), L( R2p34, R01 ) );

        X( R4,R4 ) = X( R4,R4 ) - tril( L( R4,R23p ) * ...
                 ( SkewSym_L( X( R23p,R23p ) ) * L( R4,R23p )' ) );
    otherwise
        fprintf( "urecognized version %s\n", version);
    end


end

end