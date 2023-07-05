function[] = RBKI()
    m = 10^3;
    n = 500;
    k = 500;
    A = randn(m, n/2);
    A = [A, A];
    A_cpy = A;
    p = 6;

    X_od = zeros(m, 0);
    X_ev = zeros(m, 0);
    Y_od = zeros(n, 0);
    Y_ev = zeros(n, 0);
    
    Y_i = randn(n, k);
    size(Y_i)
    for j = 1:p
        if mod(j, 2) ~= 0
            if j ~= 1
                [Y_i, ~] = qr(Y_i - (Y_od * (Y_od' * Y_i)), 0);
            end
            [Y_i, ~] = qr(Y_i, 0);
            X_i = A * Y_i;

            X_od = [X_od, X_i]; %#ok<AGROW>
            Y_od = [Y_od, Y_i]; %#ok<AGROW>

            [U, Sigma, V] = svd(X_od, 'econ', 'vector');
            V = Y_od * V;
        else
            if j ~= 2
                [X_i, ~] = qr(X_i - (X_ev * (X_ev' * X_i)), 0);
            end
            [X_i, ~] = qr(X_i, 0);
            Y_i = A' * X_i;

            X_ev = [X_ev, X_i]; %#ok<AGROW>
            Y_ev = [Y_ev, Y_i]; %#ok<AGROW>
            
            [U, Sigma, V] = svd(Y_ev', 'econ', 'vector');
            U = X_ev * U;
        end

        %fprintf("Residual norm: %e\n", sqrt(norm(U' * A - (diag(Sigma) * V'), 'fro')^2 + norm(A*V - (U * diag(Sigma)), 'fro')^2));
    end
    disp(norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
end