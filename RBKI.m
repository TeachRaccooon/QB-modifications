function[] = RBKI()
    m = 10^3;
    n = 500;
    k = 500;
    A = randn(m, n/2);
    A = [A, A];
    A_cpy = A;

    Y = randn(n, k);

    for i = 1:10
        if mod(i, 2) ~= 0
            Y_i = 
        else
            size(X)
            [X, ~] = qr(X, 0);
            Y = A' * X;
            [U_hat, Sigma, V] = svd(Y', "econ");
            U = X * U_hat;
        end

        fprintf("Residual norm: %e\n", sqrt(norm(U' * A - (Sigma * V'), 'fro')^2 + norm(A*V - (U *Sigma), 'fro')^2));
        %A = A - U * Sigma * V';
    end
    disp(norm(A_cpy - U * Sigma * V', 'fro') / norm(A_cpy, 'fro'));
end