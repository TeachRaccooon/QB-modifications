% Blocked version of algorithm 4.5 from TW2023 with a residual-based
% user-controlled stopping criteria.
function[] = RBKI_blocked(A, b_sz, tol, k, p)
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
        
        Y_i = randn(n, b_sz);

        for j = 1:min(ceil(k/b_sz), p)
            if mod(j, 2) ~= 0
                [Y_i, ~] = qr(Y_i, 0);
                % Reorthogonalization wrt odd iterates
                [Y_i, ~] = qr(Y_i - (Y_od * (Y_od' * Y_i)), 0);
                X_i = A * Y_i;
    
                X_od = [X_od, X_i]; %#ok<AGROW>
                Y_od = [Y_od, Y_i]; %#ok<AGROW>
            else
                [X_i, ~] = qr(X_i, 0);
                % Reorthogonalization wrt even iterates
                [X_i, ~] = qr(X_i - (X_ev * (X_ev' * X_i)), 0);
                Y_i = A' * X_i;
    
                X_ev = [X_ev, X_i]; %#ok<AGROW>
                Y_ev = [Y_ev, Y_i]; %#ok<AGROW>
            end
        end

        % There is no need to perform an SVD at every iteration j like it
        % is suggested in the original pseudocode. 
        if mod(p, 2) ~= 0
            [U_i, Sigma_i, V_i] = svd(X_od, 'econ', 'vector');
            V_i = Y_od * V_i;
        else
            [U_i, Sigma_i, V_i] = svd(Y_ev', 'econ', 'vector');
            U_i = X_ev * U_i;
        end

        % Output update 
        U = [U, U_i];             %#ok<AGROW>
        V = [V, V_i];             %#ok<AGROW>
        Sigma = [Sigma; Sigma_i]; %#ok<AGROW>

        % Square root of an equation 4.6 in TW2023 paper. 
        % Used for the stopping criteria here
        residual_err = sqrt(norm(U_i' * A - (diag(Sigma_i) * V_i'), 'fro')^2 + norm(A*V_i - (U_i * diag(Sigma_i)), 'fro')^2) / norm(A_cpy, 'fro');
        
        fprintf("Iteration %d, size of Sigma if terminated now: %d\n", i, nnz(Sigma));
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



