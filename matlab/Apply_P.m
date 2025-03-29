function A = Apply_P( side, p, A )

% Apply P Apply permutation to matrix from indicated side
%
% Input:  side - side from which to apply
%         p    - permutation vector
%         A    - matrix to which permutation is applied
%
% Output: A    - permuted matrix
%
% A permutation vector is a vector of integers that defines a sequence of
% permutations, which much be applied from the left or right.
% If p = [ pi_0, pi_1, ..., pi_(n-1) ], then, in order, rows or columns
% 0 and pi_0, 1 and pi_1+1, etc are swapped.
%
% copyright 2023, 2024, 2025 by Robert van de Geijn

assert( strcmp( side, 'left' ) | strcmp( side, 'right' ), ...
    'side must equal left or right' );

m = size( p, 1 );

if strcmp( side, 'left' )
    for i=1:m
        % swap ith row with i+p(i) row
        tmp = A( i, : );
        A( i, : ) = A( i+p( i, 1 ), : );
        A( i+p( i, 1 ), : ) = tmp;
    end
end

if strcmp( side, 'right' )
    for i=1:m
        % swap ith column with i+p(i) column
        tmp = A( i, : );
        tmp = A( :, i );
        A( :, i ) = A( :, i+p( i, 1 ) );
        A( :, i+p( i, 1 ) ) = tmp;
    end
end