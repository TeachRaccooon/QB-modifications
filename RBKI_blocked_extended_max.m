% Extended blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria.
% Is derived from Rob's "RBKI_blocked_extended_rob" using Max's ideas.
% I would like to call this an "iterative" approach.
function[] = RBKI_blocked_extended_max(A, b_sz, tol, k, p)
    [m, n] = size(A);
    A_cpy = A;

    U = [];
    V = [];
    Sigma = [];

    for i = 1 : ceil(k / b_sz)
        % Keeping track of prev iterates 
        X_od = zeros(m, 0);
        X_ev = zeros(m, 0);
        Y_od = zeros(n, 0);
        Y_ev = zeros(n, 0);
        Z_od = zeros(m, 0);
        Z_ev = zeros(m, 0);
        
        Y_i = randn(n, b_sz);
        Z_i = A * Y_i;

        for j = 1:p
            if mod(j, 2) ~= 0
                X_i = Z_i;
                % Reorthogonalization wrt odd iterates
                R_od = Y_i' * Y_od;
                X_i = X_i - X_od * R_od';
                Y_i = Y_i - Y_od * R_od';

                [Y_i, R] = qr(Y_i, 0);
                X_i = X_i / R;
                Z_i = A' * X_i;

                X_od = [X_od, X_i]; %#ok<AGROW>
                Y_od = [Y_od, Y_i]; %#ok<AGROW>
                Z_od = [Z_od, Z_i]; %#ok<AGROW>
            else
                Y_i = Z_i;
                % Reorthogonalization wrt even iterates
                R_ev = X_i' * X_ev;
                X_i = X_i - X_ev * R_ev';
                Y_i = Y_i - Y_ev * R_ev';

                [X_i, R] = qr(X_i, 0);
                Y_i = Y_i / R;
                Z_i = A * Y_i;
    
                X_ev = [X_ev, X_i]; %#ok<AGROW>
                Y_ev = [Y_ev, Y_i]; %#ok<AGROW>
                Z_ev = [Z_ev, Z_i]; %#ok<AGROW>
            end
        end

        % There is no need to perform an SVD at every iteration j like it
        % is suggested in the original pseudocode. 
        if mod(p, 2) ~= 0
            [U_i, Sigma_i, V_hat_i] = svd(X_od, 'econ', 'vector');
            V_i = Y_od * V_hat_i;

            residual_err = norm(Z_od * V_hat_i * diag(pinv(Sigma_i)) - V_i * diag(Sigma_i), 'fro');
        else
            [U_hat_i, Sigma_i, V_i] = svd(Y_ev', 'econ', 'vector');
            U_i = X_ev * U_hat_i;

            residual_err = norm(Z_ev * U_hat_i * diag(pinv(Sigma_i)) - U_i * diag(Sigma_i), 'fro');
        end

        % Output update 
        U = [U, U_i];             %#ok<AGROW>
        V = [V, V_i];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_i]; %#ok<AGROW>
        
        fprintf("Iteration %d, size of Sigma if terminated now: %d\n", i, nnz(Sigma));
        fprintf("Short relative residual error: %e\n", residual_err);
        fprintf("Long relative residual error: %13e\n", sqrt(norm(U_i' * A - (diag(Sigma_i) * V_i'), 'fro')^2 + norm(A*V_i - (U_i * diag(Sigma_i)), 'fro')^2));
        fprintf("||A - USigmaV'||_F / ||A||_F: %13e\n", norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro'));
        if residual_err < tol
            fprintf("RESIDUAL TERMINATION CRITERIA REACHED\n");
            break;
        end
        fprintf("\n");

        % Update A for the next iteration.
        A = A - U_i * diag(Sigma_i) * V_i';
    end
end