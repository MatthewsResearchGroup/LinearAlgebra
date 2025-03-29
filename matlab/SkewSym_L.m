function [ X ] = SkewSym_L( X )
% SkewSym_L  Skew-symmetrize X
%   Given square skew-symmmetric matrix X where only the lower triangular
%   part of the matrix is stored (with other values about the diagonal), it
%   explicitly forms the skew-symmetric matrix.
%
% Input:  X - skew-symmetric matrix with only lower triangular part stored
%
% Output: X - Skew-symmetric matrix X 
%
% copyright 2023, 2024 by Robert van de Geijn

[ m, n ] = size( X );
assert( n == m, 'X must be square' );

X = tril( X ) - tril( X )';

end