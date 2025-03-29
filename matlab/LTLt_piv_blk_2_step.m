function [ X, L, p ] = LTLt_piv_blk_2_step( X, L, nb )

% LTLt_piv_blk_Wimmer Compute permutation vector p, tridiagonal matrix T,
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
% Blocked right-looking variant with pivoting where the update has been
% modified to mimic how the 2-step algorithm updates the trailing matrix 
% with a skew-symmetric rank-2k update.
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

p = zeros( m,1 );

if m <= 2 
    return
end

% While the algorithm only requires a panel of W, the implementation
% (indexing into W) is easier when we use a full matrix

W = zeros( size( X ) );

for i=1:nb:m-1

    % Determine next block size
    ib = min( m-i+1, nb );

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; R2 = [i+1:i+ib-1]; r3=[i+ib:min(i+ib,m)];
    R4 = [i+ib+1:m];  
    R01 = [1:i]; R123 = [i:min(i+ib,m)]; R1234 = [i:m]; 
    R23 = [i+1:min(i+ib,m)]; R234 = [i+1:n]; R34 = [i+ib:m]; 
    R23skip = [ i+2:2:min(i+ib,m) ];

    % Factor next ib columns.  The elements in the first column
    % are implicitly e0
    [ X( R1234,R1234 ), L( R1234,R1234 ), p( R123, 1 ) ] = ...
        LTLt_piv_unb_left_trunc( true, X( R1234,R1234 ), ...
        L( R1234,R1234 ), p( R123,  1 ), ib );
    
    % Pivot prior columns of L
    L( R234, R01 ) = Apply_P( 'left', p( R23, 1 ), L( R234, R01 ) );

    % Split T = S - S^T
    S = Tridiag_skewsym_split( X( R23, R23 ) );

    % Form appropriate parts of W
    W( R34, R23 ) = L( R34, R23 ) * S;

    % Update x43
    X( R4,r3 )  = X( R4,r3 ) - ( W( R4,R23 ) * L( r3, R23 )' - ...
        L( R4, R23) * W( r3, R23 )' );

    % Add to last column of appropriate part of W
    W( R4,r3 ) = W( R4,r3 ) + X( R4,r3 );
    
    % Update X44, skipping columns of W known to equal zero
    X( R4,R4 ) = X( R4, R4 ) - ...
        tril( W( R4,R23skip ) * L( R4,R23skip )' - ...
        L( R4,R23skip ) * W( R4,R23skip )', -1 );
       
end

end