function S = Tridiag_skewsym_split( T )
% Tridiag_skewsym_split Split skew-symmetric tridiagonal matrix T into 
% T = S - S' where every other entry of the first subdiagonal of S equals
% zero.

[ m,n ] = size( T );

% assert( mod( m,2 ) == 0, "T must be of even size" );

S = SkewSym_L( T );

for i=1:2:n-1
    S( i+1,i ) = 0;
    if (i+2 < n )
        S( i+1,i+2 ) = 0;
    end
end


end