% Extended blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria.
% This is an algorithm, the pseudocode for which Rob has shared on
% 07/12/2023.
% I would like to call this an "incremental" approach.
function[] = RBKI_blocked_extended_rob(A, b_sz, tol, k, p)
    [m, n] = size(A);
    A_cpy = A;

    U = [];
    V = [];
    Sigma = [];

    Y_i = randn(n, b_sz);
    Z_i = A * Y_i;

    for i = 1 : ceil(n / k)
        % Keeping track of prev iterates 
        X_od = zeros(m, 0);
        X_ev = zeros(m, 0);
        Y_od = zeros(n, 0);
        Y_ev = zeros(n, 0);
        Z_od = zeros(m, 0);
        Z_ev = zeros(m, 0);
        
        
        if mod(i, 2) ~= 0
            X_i = Z_i;
            for j = 1:p
                R_od = Y_i' * Y_od;
                X_i = X_i - X_od * R_od';
                Y_i = Y_i - Y_od * R_od';
            end
            [Y_i, R_i] = qr(Y_i, 0);
            X_i = X_i * inv(R_i);

            X_od = [X_od, X_i]; %#ok<AGROW>
            Y_od = [Y_od, Y_i]; %#ok<AGROW>

            [U_i, Sigma_i, V_hat_i] = svd(X_od, 'econ', 'vector');
            V_i = Y_od * V_hat_i;
            Z_i = A' * X_i;

            Z_od = [Z_od, Z_i]; %#ok<AGROW>
            E_i = Z_od * V_hat_i * pinv(diag(Sigma_i)) - V_i * diag(Sigma_i);

            norm(E_i, 'fro')
        else
            Y_i = Z_i;
            for j = 1:p
                R_ev = X_i' * X_od;
                X_i = X_i - X_ev * R_ev';
                Y_i = Y_i - Y_ev * R_ev';
            end
            [X_i, R_i] = qr(X_i, 0);
            Y_i = Y_i * inv(R_i);

            X_ev = [X_ev, X_i]; %#ok<AGROW>
            Y_ev = [Y_ev, Y_i]; %#ok<AGROW>

            [U_hat_i, Sigma_i, V_i] = svd(Y_ev', 'econ', 'vector');
            U_i = X_ev * U_hat_i;
            Z_i = A' * Y_i;

            Z_ev = [Z_ev, Z_i]; %#ok<AGROW>

            E_i = Z_ev * U_hat_i * pinv(diag(Sigma_i)) - U_i * diag(Sigma_i);

            norm(E_i, 'fro')
        end
    end
    norm(A - U_i * diag(Sigma_i) * V_i', 'fro')
end