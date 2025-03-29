
% Test_LTLt_all Script to test all versions of LTLt algorithms using
% Test_LTLt( version, n_range, nb_range, threshold, error_only )
%
% copyright 2023, 2024 by Robert van de Geijn

n = 6;
nb = 2

rand( "seed", 1 );   % ensure the same matrix is always created.
X = rand( n, n );
L = eye( n, n );





[ T_right, L_right, p_right ] = ...
    LTLt_piv_blk( 'piv_blk_right2a', X, L, nb ); 


clc

[ T_rightb, L_rightb, p_rightb ] = ...
    LTLt_piv_blk( 'piv_blk_right2b', X, L, nb );

T_right

T_rightb

norm(T_right - T_rightb,1)

L_right

L_rightb

norm(L_right - L_rightb,1)

p_right

p_rightb

norm(p_right - p_rightb,1)


