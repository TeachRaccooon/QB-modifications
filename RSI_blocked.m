function[] = RSI_blocked(A, b_sz, tol, k, p)
    [m, n] = size(A);
    A_cpy = A;

    X = zeros(m, 0);
    Y = zeros(n, 0);
    U = [];
    Sigma = [];
    V = [];
    for i = 1: ceil(k / b_sz)
        Y_i = randn(n, b_sz);
        for j = 1:p
            if mod(j, 2) ~= 0
                Y_i = orth(Y_i);
                Y_i = orth(Y_i - (Y * (Y' * Y_i)));
                X_i = A * Y_i;
                [U_i, Sigma_i, V_i] = svd(X_i, 'econ', 'vector');
                V_i = Y_i * V_i;
            else
                X_i = orth(X_i);
                X_i = orth(X_i - (X * (X' * X_i)));
                Y_i = A' * X_i;
                [U_i, Sigma_i, V_i] = svd(Y_i', 'econ', 'vector');
                U_i = X_i * U_i;
            end
        end
        X = [X, X_i];             %#ok<AGROW>
        Y = [Y, Y_i];             %#ok<AGROW>
        U = [U, U_i];             %#ok<AGROW>
        V = [V, V_i];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_i]; %#ok<AGROW>

        residual_err = sqrt(norm(U_i' * A - (diag(Sigma_i) * V_i'), 'fro')^2 + norm(A*V_i - (U_i * diag(Sigma_i)), 'fro')^2) / norm(A_cpy, 'fro');
        
        fprintf("Iteration %d, rank if terminated now %d\n", i, size(U, 2));
        fprintf("Residual error: %e\n\n", residual_err);

        if residual_err < tol
            fprintf("RESIDUAL TERMINATION CRITERIA REACHED\n");
            break;
        end

        A = A - U_i * diag(Sigma_i) * V_i';
    end

    fprintf("Final error ||A - USigmaV'||_F / ||A||_F: %e\n", norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
end