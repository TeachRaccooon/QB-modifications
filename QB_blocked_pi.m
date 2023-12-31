function [Q, B] = QB_blocked_pi(A, block_size, tol, k, p)
    A_cpy = A;
    indicator_fro = 0;
    indicator_spec = 0;
    norm_A = normest(A, 'fro');
    norm2_A = normest(A, 2);

    % Setting initial error to zero.
    approximation_error = 0;
    [m, n] = size(A);
    norm_B = 0;
    % Pre-initialization of output matrices.
    Q = zeros(m, 0);
    B = zeros(0, n);
    % Iterative stage.
    for i = 1 : ceil(k / block_size)
        %fprintf(“K in QB %d\n”, k);
        if size(B, 1) + block_size > k
            block_size = k - size(B, 1);
        end
        % Consstructiong a sketch for current iteration.
        Q_i = RSI(A, block_size, p);
        % Ensuring orthogonalization of Q.
        Q_i = orth(Q_i - (Q * (Q' * Q_i)));
        % Deterministic computation of B_i.
        B_i = Q_i' * A;
        % Approximation error check.
        norm_B = hypot(norm_B, norm(B_i, 'fro'));
        prev_error = approximation_error;
        approximation_error = sqrt((norm_A - norm_B) * (norm_A + norm_B)) / norm_A;
        % Handling the round-off error accumulation.
        if (i > 1) && (approximation_error > prev_error)
            break
        end
        % Output update.
        Q = [Q, Q_i]; %#ok<AGROW>
        B = [B; B_i]; %#ok<AGROW>
        fprintf("Iteration %d, rank if terminated now %d\n", i, size(Q, 2));
        fprintf("sqrt(||A||_F^2 - ||B||_F^2) / ||A||_F^2: %e\n", approximation_error);
        fprintf("||A - QB||_F / ||A||_F %e\n", norm(A_cpy - Q*B, 'fro') / normest(A_cpy, 'fro'));
        fprintf("||A - QB||_2 / ||A||_2 %e\n", norm(A_cpy - Q*B, 2) / normest(A_cpy, 2));
        fprintf("||B_i||_2 /||A||_2: %e\n", norm(B_i, 2) / norm2_A);
        fprintf("||B_i||_2 / ||Delta_i-1||_2: %e\n", norm(B_i, 2) / normest(A, 2));


        if approximation_error < tol && ~indicator_fro
            fprintf("FRO TERMINATION CRITERIA REACHED AND COMPUTED AT RANK %d, iteration %d\n", block_size * i, i);
            indicator_fro = 1;
            %break;
        elseif approximation_error < tol && indicator_fro
            fprintf("FRO TERMINATION CRITERIA REACHED PREVIOUSLY\n");
        end
        
        if norm(B_i, 2) / norm2_A < tol && ~indicator_spec
            fprintf("SPEC TERMINATION CRITERIA REACHED AT RANK %d, COMPUTED AT ITERATION %d\n", block_size * (i - 1), i);
            indicator_spec = 1;
            %break;
        elseif norm(B_i, 2) / norm2_A < tol && indicator_spec
            fprintf("SPEC TERMINATION CRITERIA REACHED PREVIOUSLY\n");
        end

        fprintf("\n");

        A = A - Q_i * B_i;
    end
end

function [Q] = RSI(A, k, p)
    [m, n] = size(A);
    v = 2 * p + 1;

    % Odd number of passes over A.
    if(mod(v, 2) == 0)
        Omega = randn(m, k);
        
        if (v > 2)
            [Q, ~] = lu(A' * Omega);
        else
            [Q, ~] = qr(A' * Omega, 0);
        end
    % Even number of passes over A.
    else
        Q = randn(n, k);
        if p == 0
            [Q, ~] = qr(A * Q, 0);
        end
    end

    for i = 1 : p
        [Q, ~] = lu(A * Q);
        [Q, ~] = lu(A' * Q);
    end

    [Q, ~] = qr(A * Q, 0);
end