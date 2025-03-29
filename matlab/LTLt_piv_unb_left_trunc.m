function [ X, L, p ] = LTLt_piv_unb_left_trunc( impl_e0, X, L, p, k )

% LTLt_piv_unb_left_trunc Compute permutation vector p, tridiagonal matrix 
% T and unit lower triangular matrix L such that
%
%         P(p) X P(p)' = L * T * L', where 
%
% Truncated version: only first k columns of X and k+1 columns of L are 
% computed.
%
% Input:  impl_e0 - true if first column of L is implicitly e_0, false 
%                   otherwise (use stored column)
%         X - skew-symmetric matrix, stored in the strictly lower
%             triangular part.
%         L - Lower triangular matrix L passed in as identity matrix.
% .       k - only update first k columns of X and k+1 columns of L.
%         p - pivot vector (passed in since first entry created by previous
%             iteration of the blocked algorithm.)
%
% Output: X - Tridiagonal skew-symmetric matrix T, stored in the strictly
%             lower triangular part.  Upper triangular part is left
%             untouched.
%         L - Unit lower triangular matrix.
%         p - pivot vector.
%
% Unblocked left-looking variant with pivoting
%
% copyright 2023, 2024 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, "X must be square" );

if m <= 2
    return
end

if m > 1 & impl_e0
    % Store first column of L and set elements below diagonal to 0 
    Lfirst = L( :, 1 );
    L( 2:m, 1 ) = zeros( m-1, 1 );
end


before = [ X L ];

for i=1:min(k,m-1)
    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; R3 = [i+2:m]; 
    R01 = [1:i]; R23= [i+1:m];
    
    % / chi21 \ -:= / l20' lambda21 \ / X00 -x10' \ / l10 \
    % \  x31  /     \ L30     l31   / \ x10   0   / \  1  /
    X( R23, r1 ) = X( R23,r1 ) - ...
        L( R23, R01 ) * ( SkewSym_L( X( R01, R01 ) ) * L( r1, R01 )' );

    % Determine which row to pivot to top    
    p( r2, 1 ) = Iamax( X( R23, r1 ) );

    % Pivot current column
    X( R23,r1 ) = Apply_P( 'left', p( r2, 1 ), X( R23,r1 ) );

    % Compute next column of L
    L( R3, r2 ) = X( R3, r1 ) / X( r2, r1 );
    
    % Pivot rest of X
    X( R23,R23 ) = SkewSymm_Apply_P( p( r2, 1 ), X( R23, R23 ) );

    % Pivot L
    Lold= L( R23, R01 );
    L( R23, R01 ) = Apply_P( 'left', p( r2, 1 ), L( R23, R01 ) );
   
    % x31 := 0
    X( R3, r1 ) = zeros( size( R3,1 ), 1 );
   
end

if m > 1 & impl_e0
    % Restore first column.  This column will be pivoted within the
    % blocked algorithm
    L( :, 1 ) = Lfirst;
end

end
