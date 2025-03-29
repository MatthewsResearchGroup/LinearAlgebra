function [ X ] = SkewSymm_Apply_P( p, X )

% Symm_Apply_P: Apply sequence of permutations defined by p to 
% skew-symmetric matrix X, starting indexing at 0
%
%  Input: p - permutation vector
%         X - Skew-symmetric matrix to be permuted
%
%  Output: X - permuted matrix
%
% A permutation vector is a vector of integers that defines a sequence of
% permutations, which much be applied from the left and the right.
% If p = [ pi_0, pi_1, ..., pi_(n-1) ], then, in order, rows and columns
% 0 and pi_0, 1 and pi_1+1, etc are swapped.
%
% Remember that in thi workd indexing starts at zero, and Matlab starts 
% indexing at one, requiring the appropriate adjustments.
%
% Only the strictly lower triangular part of the skew-symmetric matrix is 
% stored, meaning that careful attention must be paid to which parts of 
% rows and columns are swapped, and what parts need to be negated.
%
% copyright 2023, 2024 by Robert van de Geijn

% Extract how many permutations are to be performed.
m = size( p, 1 );

if m == 0
    return
end

% Quick and dirty implementation meant to be understandable rather than
% fast

% Backup upper triangular part
Temp = triu( X );

% Skew-symmetrize X
X = SkewSym_L( X );

% Apply permutations from left
X = Apply_P( 'left', p, X );

% Apply permutations from right
X = Apply_P( 'right', p, X );

% Restore upper triangular part
X = tril( X, -1 ) + Temp;

end