function[] = RBKI()
    m = 10^3;
    n = 500;
    k = 100;
    A = randn(m, n/2);
    A = [A, A];
    A_cpy = A;
    p = 6;

    X_od = zeros(m, 0);
    X_ev = zeros(m, 0);
    Y_od = zeros(n, 0);
    Y_ev = zeros(n, 0);
    
    Y_i = randn(n, k);
    for j = 1:min(ceil(n/k), p)
        if mod(j, 2) ~= 0
            [Y_i, ~] = qr(Y_i, 0);
            [Y_i, ~] = qr(Y_i - (Y_od * (Y_od' * Y_i)), 0);
            X_i = A * Y_i;

            X_od = [X_od, X_i]; %#ok<AGROW>
            Y_od = [Y_od, Y_i]; %#ok<AGROW>

            [U, Sigma, V_hat] = svd(X_od, 'econ', 'vector');
            V = Y_od * V_hat;

            norm(A' * X_od * V_hat * pinv(diag(Sigma)) - V * diag(Sigma), 'fro');
        else
            [X_i, ~] = qr(X_i, 0);
            [X_i, ~] = qr(X_i - (X_ev * (X_ev' * X_i)), 0);
            Y_i = A' * X_i;

            X_ev = [X_ev, X_i]; %#ok<AGROW>
            Y_ev = [Y_ev, Y_i]; %#ok<AGROW>
            
            [U_hat, Sigma, V] = svd(Y_ev', 'econ', 'vector');
            U = X_ev * U_hat;

            norm(A * Y_ev * U_hat * pinv(diag(Sigma)) - U * diag(Sigma), 'fro');
        end

        %fprintf("Residual norm: %e\n", sqrt(norm(U' * A - (diag(Sigma) * V'), 'fro')^2 + norm(A*V - (U * diag(Sigma)), 'fro')^2));
        disp(norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
    end
end