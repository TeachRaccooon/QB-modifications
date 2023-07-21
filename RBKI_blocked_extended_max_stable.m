% Extended blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria + it is stable.
% This is an algorithm that is based on Rob's presudocode from
% 07/18/2023, reworked by Max
function[] = RBKI_blocked_extended_rob_stable(A, block_size, inner_block_size_factor, tol, target_rank, num_iters)
    [m, n] = size(A);
    A_cpy = A;
    [U_true, Sigma_true, V_true] = svd(A_cpy, 'econ', 'matrix');

    U     = [];
    Sigma = [];
    V     = [];

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
                [Y_i, R_ii] = qr(Y_i, 0);
                R = [R, R_i; zeros(size(R_ii, 1), size(R, 2)), R_ii]; %#ok<AGROW>
                Z_i = A * Y_i;
                X_i = Z_i;
    
                Y_od = [Y_od, Y_i]; %#ok<AGROW>
                Z_od = [Z_od, Z_i]; %#ok<AGROW>
            else
                S_i = X_ev' * X_i;
                X_i = X_i - X_ev * S_i;
                [X_i, S_ii] = qr(X_i, 0);
                S = [S, S_i; zeros(size(S_ii, 1), size(S, 2)), S_ii]; %#ok<AGROW>
                W_i = A' * X_i;
                Y_i = W_i;
    
                X_ev = [X_ev, X_i]; %#ok<AGROW>
                W_ev = [W_ev, W_i]; %#ok<AGROW>
            end
            if mod(i, 2) ~= 0
                [U_hat_j, Sigma_j, V_hat_j] = svd(R', 'econ', 'vector');
                U_j = X_ev * U_hat_j;
                V_j = Y_od * V_hat_j;
                E = Z_od * V_hat_j - U_j * diag(Sigma_j);

                E_true = A_cpy * V_j - U_j * Sigma_j;
            else
                [U_hat_j, Sigma_j, V_hat_j] = svd(S, 'econ', 'vector');
                U_j = X_ev * U_hat_j;
                V_j = Y_od * V_hat_j;
                E = W_ev * U_hat_j - V_j * diag(Sigma_j);
                
                E_true = A_cpy * U_j - V_j * Sigma_j;
            end

                residual_err = norm(E(:, 1:block_size), 'fro');
                fprintf("Short residual error: %e, Krylov iterations: %d\n", residual_err, i);
                fprintf("True residual error: %e, Krylov iterations: %d\n", norm(E_true(:, 1:block_size), 'fro'), i);
                fprintf("RangeFinder check: %e\n", norm(A - (U_j * U_j' * A), 'fro') / norm(A, 'fro'));
                fprintf("Orth check: %e\n", norm(U_j' * U_j - eye(size(U_j, 2), size(U_j, 2)), 'fro'));
        end

        % Output update 
        U = [U, U_j];             %#ok<AGROW>
        V = [V, V_j];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_j]; %#ok<AGROW>

        %U = [U, U_j(:, 1:block_size)];             %#ok<AGROW>
        %V = [V, V_j(:, 1:block_size)];             %#ok<AGROW>
        %Sigma = [Sigma; Sigma_j(1:block_size, :)]; %#ok<AGROW>

        size(Sigma)
        sz = min(inner_block_size_factor * j * ceil(num_iters / 2) * block_size, n);

        fprintf("Main loop iteration %d\n", j);
        fro_err = norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro');
        %fro_err = norm(A_cpy - U(:, 1:(i * block_size)) * diag(Sigma(1:(i*block_size), :)) * V(:, 1:(i*block_size))', 'fro') / norm(A_cpy, 'fro');
        fprintf("||A - USigmaV'||_F / ||A||_F: %13e\n", fro_err);
        fprintf("True approximation %e\n", norm(A_cpy - U_true(:, 1:sz) * Sigma_true(1:sz, 1:sz) * V_true(:, 1:sz)', 'fro') / norm(A_cpy, 'fro'));

        if fro_err < tol
            break;
        end

        % Update A for the next iteration.
        A = A - (U_j * diag(Sigma_j) * V_j');
        fprintf("\n");
    end
    size(Sigma)
    rank(U * diag(Sigma) * V')
    disp(norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'))
end