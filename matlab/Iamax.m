function [ pi ] = Iamax ( x )

% Iamax Return index of element in x with largest magnitude (starting 
% indexing at 0)
%
% Input
%   x -  Column vector 
%
% Output
%   pi - index of element in x with largest magnitude

[ xmax, pi ] = max( abs( x ) );

pi = pi-1;

end