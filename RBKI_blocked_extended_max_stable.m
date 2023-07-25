% Extended blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria + it is stable.
% This is an algorithm that is based on Rob's presudocode from
% 07/18/2023, reworked by Max
function[] = RBKI_blocked_extended_rob_stable(A, block_size, inner_block_size_factor, tol, target_rank, num_iters)
    [m, n] = size(A);
    A_cpy = A;
    [U_true, Sigma_true, V_true] = svd(A_cpy, 'econ', 'matrix');

    U     = zeros(m, 0);
    Sigma = [];
    V     = zeros(n, 0);

    for j = 1:ceil(target_rank / block_size)
        Y_i = randn(n, inner_block_size_factor * block_size);
        Z_i = A * Y_i;
        [X_i, ~] = qr(Z_i, 0);
        W_i = A' * X_i;
        Y_i = W_i;
    
        R = [];
        S = [];
        
        X_ev = X_i;
        Y_od = zeros(n, 0);
        
        Z_od = [];
        W_ev = W_i;

        if (j > 1)
            %num_iters = 2;
        end

        for i = 1:num_iters
            if mod(i, 2) ~= 0
                R_i = Y_od' * Y_i;
                Y_i = Y_i - Y_od * R_i;
                update = Y_od' * Y_i;
                Y_i = Y_i - Y_od * update;
                [Y_i, R_ii] = qr(Y_i, 0);
                R = [R, R_i; zeros(size(R_ii, 1), size(R, 2)), R_ii]; %#ok<AGROW>
                Z_i = A * Y_i;
                X_i = Z_i;
    
                Y_od = [Y_od, Y_i]; %#ok<AGROW>
                Z_od = [Z_od, Z_i]; %#ok<AGROW>
    
                [U_hat, Sigma_j, V_hat] = svd(R', 'econ', 'vector');
                U_j = X_ev * U_hat;
                V_j = Y_od * V_hat;
                
                E = (Z_od * V_hat) - (U_j * diag(Sigma_j));
                E_true = (A * V_j) - (U_j * diag(Sigma_j));
            else
                S_i = X_ev' * X_i;
                X_i = X_i - X_ev * S_i;
                update = X_ev' * X_i;
                X_i = X_i - X_ev * update;
                [X_i, S_ii] = qr(X_i, 0);
                S = [S, S_i; zeros(size(S_ii, 1), size(S, 2)), S_ii]; %#ok<AGROW>
                W_i = A' * X_i;
                Y_i = W_i;
    
                X_ev = [X_ev, X_i]; %#ok<AGROW>
                W_ev = [W_ev, W_i]; %#ok<AGROW>
    
                [U_hat, Sigma_j, V_hat] = svd(S, 'econ', 'vector');
                U_j = X_ev * U_hat;
                V_j = Y_od * V_hat;
                E = (W_ev * U_hat) - (V_j * diag(Sigma_j));
    
                E_true = (A' * U_j) - (V_j * diag(Sigma_j));
            end
            
            residual_err = norm(E(:, 1:block_size), 'fro');
            true_residual_err = norm(E_true(:, 1:block_size), 'fro');
            fprintf("Short residual error: %e, iteration: %d\n", residual_err, i);
            fprintf("True residual error:  %e, iteration: %d\n", true_residual_err, i);
            if residual_err < tol
               break;
            end
        end
        %[U_j, ~] = qr(U_j - (U * (U' * U_j)), 0);
        %[V_j, ~] = qr(V_j - (V * (V' * V_j)), 0);

        % Output update 
        U = [U, U_j];             %#ok<AGROW>
        V = [V, V_j];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_j]; %#ok<AGROW>

        sz = min(inner_block_size_factor * j * ceil(num_iters / 2) * block_size, n);
        fprintf("Main loop iteration %d\n", j);
        fro_err = norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro');
        fprintf("||A - USigmaV'||_F / ||A||_F: %13e\n", fro_err);

        fprintf("\n\n");
        if fro_err < tol
            break;
        end

        % Update A for the next iteration.
        A = A - (U_j * diag(Sigma_j) * V_j');
    end
    fprintf("Size of the final Sigma: %d\n", size(Sigma, 1));
    fprintf("Rank of the found factorization: %d \n", rank(U * diag(Sigma) * V'));
    fprintf("Relative Fro norm error: %e\n", norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
    norm((A_cpy' * U) - (V * diag(Sigma)))
    norm((A_cpy * V) - (U * diag(Sigma)))
    norm(U' * U - eye(size(U, 2), size(U, 2)), 'fro')
    norm(V' * V - eye(size(V, 2), size(V, 2)), 'fro')
end