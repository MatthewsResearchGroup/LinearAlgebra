function [ X, L, p ] = LTLt_unb_part( impl_e0, X, L, k )

% Compute tridiagonal matrix T and unit lower triangular matrix L such that
%         X = L * T * L' 
%
% Input:  impl_e0 - 'true':  First column of L is implicitly 0
%                   'false': First column is as passed in
%         X - skew-symmetric matrix
%         L - passed in as identity matrix
%         k - only factor first k columns
%
% Output: X - Tridiagonal skew-symmetric matrix T
%         L - Unit lower triangular matrix.  
%
% Unblocked left-looking variant ,s the first k columns of
% T and first (k+1) columns of L, leaving the remainder untouched.
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, "X must be square" );

if m > 1 & impl_e0
    % Store first column of L and set elements below diagonal to 0 
    Lfirst = L( :, 1 );
    L( 2:m, 1 ) = zeros( m-1, 1 );
end

for i=1:min(k,m-1)

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; r2 = [i+1:i+1]; R3 = [i+2:n]; 
    R01 = [1:i]; R23= [i+1:n];
    
    % / chi21 \ -:= / l20' lambda21 \ / X00 -x10' \ / l10 \
    % \  x31  /     \ L30     l31   / \ x10   0   / \  1  /
    X( R23, r1 ) = X( R23,r1 ) - ...
        L( R23, R01 ) * ( SkewSym_L( X( R01, R01 ) ) * L( r1, R01 )' );

    % l32 := x31 / chi21
    L( R3, r2 ) = X( R3, r1 ) / X( r2, r1 );

    % x31 := 0
    X( R3, r1 ) = zeros( size( R3,1 ), 1 );
   
end

if m > 1 & impl_e0
    % Restore first colum
    L( :, 1 ) = Lfirst;
end

end
