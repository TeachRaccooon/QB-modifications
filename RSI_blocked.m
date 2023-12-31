% Blocked version of algorithm 4.3 from TW2023 with a residual-based
% user-controlled stopping criteria.
function[] = RSI_blocked(A, b_sz, tol, k, p)
    [m, n] = size(A);
    A_cpy = A;

    X = zeros(m, 0);
    Y = zeros(n, 0);
    U = [];
    V = [];
    Sigma = [];

    for i = 1 : ceil(k / b_sz)
        Y_i = randn(n, b_sz);
        for j = 1:p
            if mod(j, 2) ~= 0
                Y_i = orth(Y_i);
                % Reorthogonalization wrt previous iterates
                Y_i = orth(Y_i - (Y * (Y' * Y_i)));
                X_i = A * Y_i;
            else
                X_i = orth(X_i);
                % Reorthogonalization wrt previous iterates
                X_i = orth(X_i - (X * (X' * X_i)));
                Y_i = A' * X_i;
            end
        end
        
        % There is no need to perform an SVD at every iteration j like it
        % is suggested in the original pseudocode. 
        if mod(p, 2) ~= 0
            [U_i, Sigma_i, V_i] = svd(X_i, 'econ', 'vector');
            V_i = Y_i * V_i;
        else
            [U_i, Sigma_i, V_i] = svd(Y_i', 'econ', 'vector');
            U_i = X_i * U_i;
        end

        % Output update 
        X = [X, X_i];             %#ok<AGROW>
        Y = [Y, Y_i];             %#ok<AGROW>
        U = [U, U_i];             %#ok<AGROW>
        V = [V, V_i];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_i]; %#ok<AGROW>

        % Square root of an equation 4.6 in TW2023 paper. 
        % Used for the stopping criteria here
        residual_err = sqrt(norm(U_i' * A - (diag(Sigma_i) * V_i'), 'fro')^2 + norm(A*V_i - (U_i * diag(Sigma_i)), 'fro')^2) / norm(A_cpy, 'fro');
        
        fprintf("Iteration %d, rank if terminated now: %d\n", i, nnz(Sigma));
        fprintf("Relative residual error: %17e\n", residual_err);
        fprintf("||A - USigmaV'||_F / ||A||_F: %e\n", norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));

        if residual_err < tol
            fprintf("RESIDUAL TERMINATION CRITERIA REACHED\n");
            break;
        end
        fprintf("\n");

        % Update A for the next iteration.
        A = A - U_i * diag(Sigma_i) * V_i';
    end
end