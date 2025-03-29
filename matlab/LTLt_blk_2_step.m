function [ X, L ] = LTLt_blk_2_step( X, L, nb )

% LTLt_blk_2_step Compute tridiagonal matrix T and unit lower triangular 
% matrix L such that
%
%         X  = L * T * L', where 
%
% Input:  X - skew-symmetric matrix, stored in the strictly lower
%             triangular part.
%         L - Lower triangular matrix L passed in as identity matrix. 
%         nb - block size to be used.
%
% Output: X - Tridiagonal skew-symmetric matrix T, stored in the strictly
%             lower triangular part.  Upper triangular part is left
%             untouched.
%         L - Unit lower triangular matrix.
%
% Blocked 2-step algorithm that accumulates skew-symmetric rank-2k updates
%
% Notice that not all of W needs to be kept
%
% copyright 2023, 2024, 2025 by Robert van de Geijn


[ m, n ] = size( X );
assert( n == m, 'X must be square' );

W = zeros( size( L ) );

for i=1:nb:m-2
    % Determine next block size
    ib = min( m-i+1, nb );

    % Set the various ranges
    R0 = [1:i-1]; r1 = [i:i]; R2 = [i+1:i+ib-1]; r3=[i+ib:i+ib];
    R4 = [i+ib+1:m];  R1234 = [i:m]; R23 = [i+1:i+ib]; R34 = [i+ib:m];
    R23skip = [ i+2:2:i+ib ];

    % Factor next kb columns.  The elements in the first column
    % are implicitly e0
    [ X( R1234,R1234 ), L( R1234,R1234 ), W( R1234,R1234 ) ] = ...
        LTLt_unb_2_step_trunc( X( R1234,R1234 ), L( R1234,R1234 ), ...
        W( R1234,R1234 ), ib );

    if size( R4,2 ) > 0
        X( R4, R4 ) = X( R4, R4 ) + ...
            tril( L( R4,R23skip ) * W( R4,R23skip )' ...
            - W( R4,R23skip ) * L( R4,R23skip )', -1 );
    end
    
end

end