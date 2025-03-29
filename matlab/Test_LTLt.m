function n_errors = Test_LTLt( version, n_range, nb_range, threshold, ...
    error_only )
% Test_LTLt Test various implementations for computing the LTLt
% factorization.
% 
% Input:  version    - Version to be tested
%         n_range    - Range of problem sizes to be tested
%         nb_range   - Range of block sizes to be tested              
%         threshold  - Threshold where norm of residual becomes suspect
%                     (a value in the 1e-10 range is a reasonable choice.)
%         error_only - Only report information when there is an error
%
% copyright 2023, 2024, 2025 by Robert van de Geijn


n_errors = 0;

disp( sprintf( 'Testing %s', version ) );
disp( '  n  nb         residual');
disp( '----------------------------');

for nb=nb_range

    assert( ~( ( strcmp( version, 'blk_Wimmer' ) | ...
                 strcmp( version, 'piv_blk_Wimmer' ) ) & ...
                 mod( nb, 2 ) ), ...
              'For Wimmer s algorithms nb must be even' );

    for n=n_range
        % create a random matrix.  The upper triangular part of the matrix 
        % will not be computed with so the skew-symmetric matrix is 
        % defined by what is stored in the strictly lower triangular part 
        % of X
        rand( "seed", 1 );   % ensure the same matrix is always created.
        X = rand( n, n );
        L = eye( n, n );

        switch version
            case 'unb_right'
                [ T, L ] = LTLt_unb( 'unb_right', X, L );
            case 'unb_left'
                [ T, L ] = LTLt_unb( 'unb_left', X, L );
            case 'unb_2_step'
                [ T, L ] = LTLt_unb_2_step( X, L );
            case 'blk_right'
                [ T, L ] = LTLt_blk( 'blk_right', X, L, nb );
            case 'blk_fused_righta'
                [ T, L ] = LTLt_blk( 'blk_fused_righta', X, L, nb );
            case 'blk_fused_rightb'
                [ T, L ] = LTLt_blk( 'blk_fused_rightb', X, L, nb );
            case 'blk_left'
                [ T, L ] = LTLt_blk( 'blk_left', X, L, nb );
            case 'blk_2_step'
                [ T, L ] = LTLt_blk_2_step( X, L, nb );
            case 'piv_unb_right'
                [ T, L, p ] = LTLt_piv_unb( 'piv_unb_right', X, L );
            case 'piv_unb_left'
                [ T, L, p ] = LTLt_piv_unb( 'piv_unb_left', X, L );
            case 'piv_unb_2_step'
                [ T, L, p ] = LTLt_piv_unb_2_step( X, L );
            case 'piv_blk_right'
                [ T, L, p ] = LTLt_piv_blk( 'piv_blk_right', X, L, nb ); 
            case 'piv_blk_right2a'
                [ T, L, p ] = LTLt_piv_blk( 'piv_blk_right2a', X, ...
                    L, nb );
            case 'piv_blk_right2b'
                [ T, L, p ] = LTLt_piv_blk( 'piv_blk_right2b', X, ...
                    L, nb );
            case 'piv_blk_2_step'

                [ T, L, p ] = LTLt_piv_blk_2_step( X, L, nb );
            otherwise
                fprintf( "urecognized version %s\n", version);
        end

        if ~strcmp( version(1:3), 'piv' )
            % no pivoting so set pivot vector to zeros for error checking
            p = zeros( n, 1 );
        end

        % Report result by checking P * X * P' - L * T * L' 
        % Report warning if norm is larger than 1.0e-07.  There is a small
        % chance that due to the problem being numerically ill-conditioned
        % a large residual is detected, so this is not a fool-proof test.
        residual = norm( tril( SkewSymm_Apply_P( p, X ), -1 ) - ...
            tril( L * SkewSym_L( T ) * L', -1 ), 1 );

        if residual > threshold
            warning = 'alert: possible error in residual\n';
        else
            warning = '';
        end

        % If pivoting, check that L is lower triangular and all entries 
        % have absolute value less than or equal to 1
        if strcmp( version(1:3), 'piv' ) & max( max( tril( L ) ) ) > 1
            warning = strcat( warning, ...
                'magnitude of entries of L must be less than one\n' );
        end

        % Check that strictly lower triangular part of T is tridiagonal
        if max( max( abs( tril( T, -2 ) ) ) ) ~= 0
            warning = strcat( warning, ...
               'tril( T ) is not tridiagonal\n' );
        end

        % Check that upper triangular part of T equals original X
        if max( max( abs( triu( X, 1 ) - triu( T, 1 ) ) ) ) ~= 0
            warning = strcat( warning, ...
                'upper triangular part of X/T corrupted' );
        end

        % If approppriate, report result
        if ( ~error_only | ~strcmp( warning, '' ) )
            disp( sprintf(['%' ...
                '3d %3d          %4.1e %s'], n, nb, residual, ...
                warning ) );
        end

        if ~strcmp( warning, '' )
            n_errors = n_errors+1;
        end

    end

end

disp( sprintf( 'Number of potential problems detected for %s: %4d', ...
    version, n_errors ) );

n_errors

end