% Extended blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria + it is stable.
% This is an algorithm, the pseudocode for which Rob has shared on
% 07/18/2023.
% I would like to call this an "incremental" approach.
function[] = RBKI_blocked_extended_rob_stable(A, k, tol, r, num_iters)
    [~, n] = size(A);

    Y_i = randn(n, k);
    Z_i = A * Y_i;
    [X_i, ~] = qr(Z_i, 0);
    W_i = A' * X_i;
    Y_i = W_i;

    R = []; S = [];
    
    X_ev = X_i; Y_od = zeros(n, 0);
    
    Z_od = []; W_ev = W_i;

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

            [U_hat, Sigma, V_hat] = svd(R', 'econ', 'vector');
            U = X_ev * U_hat;
            V = Y_od * V_hat;
            
            E = (Z_od * V_hat) - (U * diag(Sigma));
            E_true = (A * V) - (U * diag(Sigma));
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

            [U_hat, Sigma, V_hat] = svd(S, 'econ', 'vector');
            U = X_ev * U_hat;
            V = Y_od * V_hat;
            E = (W_ev * U_hat) - (V * diag(Sigma));

            E_true = (A' * U) - (V * diag(Sigma));
        end
        
        residual_err = norm(E(:, 1:r), 'fro');
        true_residual_err = norm(E_true(:, 1:r), 'fro');
        fprintf("Short residual error: %e, iteration: %d\n", residual_err, i);
        fprintf("True residual error:  %e, iteration: %d\n\n", true_residual_err, i);
        if residual_err < tol
           % break;
        end
    end
    %fro_err = norm(A - U * diag(Sigma) * V', 'fro') / norm(A, 'fro');
    %fprintf("||A - USigmaV'||_F / ||A||_F: %13e\n", fro_err);
end