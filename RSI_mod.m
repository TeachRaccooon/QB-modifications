function[] = RSI_mod()
    m = 10^3;
    n = 500
    k = 500;
    A = randn(m, n/2);
    A = [A, A];
    b_sz = 50;
    tol = 1e-15;
    p = 3;
    A_cpy = A;
    
    Q = zeros(m, 0);
    U = [];
    Sigma = [];
    V = [];
    for i = 1: ceil(k / b_sz)

        Q_i = randn(n, b_sz);
        
        Q_i = orth(A * Q_i);

        Q_i = orth(Q_i - (Q * (Q' * Q_i)));
        %B_i = Q_i' * A;
        [U_i, Sigma_i, V_i] = svd(Q_i' * A, 'econ', 'vector');
        U_i = Q_i * U_i;

        Q = [Q, Q_i]; %#ok<AGROW>
        U = [U, U_i]; %#ok<AGROW>
        Sigma = [Sigma; Sigma_i]; %#ok<AGROW>
        V = [V, V_i]; %#ok<AGROW>

        fprintf("Residual norm: %e\n", sqrt(norm(U_i' * A - (diag(Sigma_i) * V_i'), 'fro')^2 + norm(A*V_i - (U_i * diag(Sigma_i)), 'fro')^2));

        A = A - U_i * diag(Sigma_i) * V_i';
    end

    disp(norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
end