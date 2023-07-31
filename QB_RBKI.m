function[err_ratio_entry, rel_fro_err] = QB_RBKI(A, inner_block_size_factor, target_rank, num_iters)
    [~, n] = size(A);
    A_cpy = A;
    [U_true, Sigma_true, V_true] = svd(A_cpy, 'econ', 'matrix');
    
    Y_i = randn(n, inner_block_size_factor * target_rank);
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
        end
    end
    if mod(num_iters, 2) ~= 0
        %[U_hat, ~] = qr(R', 0);
        %U = X_ev * U_hat;
        [U_hat, ~, ~] = svd(R', 'econ', 'vector');
        U = X_ev * U_hat;
    else
        %[U_hat, ~] = qr(S, 0);
        %U = X_ev * U_hat;
        [U_hat, ~, ~] = svd(S, 'econ', 'vector');
        U = X_ev * U_hat;
    end

    Q = U(:, 1:target_rank);
    B = Q' * A;

    %size(Q)
    %size(B)

    fro_err = norm(A_cpy - Q * B, 'fro');
    rel_fro_err = fro_err / norm(A_cpy, 'fro');
    curr_sz = size(Q, 2);
    true_err = norm(A_cpy - U_true(:, 1:curr_sz) * Sigma_true(1:curr_sz, 1:curr_sz) * V_true(:, 1:curr_sz)', "fro");
    err_ratio_entry = (fro_err/ true_err) - 1;

    fprintf("Rel Fro Error: %e\n", rel_fro_err);
    fprintf("Rel True Error: %e\n", true_err / norm(A_cpy, 'fro'));
    
    fprintf("Fro Error / True Fro Error - 1: %e\n", err_ratio_entry); 

    %fprintf("Final orthogonality of Q: %e\n", norm(Q' * Q - eye(size(Q, 2), size(Q, 2)), 'fro'));
    %fprintf("Inner dimension of Q, B: %d\n", size(Q, 2));
end