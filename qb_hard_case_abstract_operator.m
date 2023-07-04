function[] = qb_hard_case_abstract_operator()

    
    n = 10^5;
    A = zeros(1, n);
    A(1:500) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    A(1, 1) = 10;
    A_cpy = A;
    b_sz = 50;
    k = 600;
    tol = 1e-15;
    p = 1;
    A = {A};
%{
    n = 10;
    A = zeros(1, n);
    A(1:7) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    b_sz = 3;
    k = 10;
    p = 0;
    tol = 1e-15;
    %cell array
    A = {A};
    A_cpy = A;
%}
    qb_2(A, b_sz, tol, k, p, A_cpy);
end

function [Q, B] = qb_2(A, block_size, tol, k, p, A_cpy)
    indicator_fro = 0;
    indicator_spec = 0;
    norm_A = normest(A{1}, 'fro');
    norm2_A = normest(A{1}, 2);
    % Early termination check on an empty input.
    if norm_A == 0
        fprintf('The input matrix is empty.');
        return
    end
    % Setting initial error to zero.
    approximation_error = 0;
    [m, n] = size(A{1});
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
        %s = RandStream("dsfmt19937");
        %[Q_i, ~] = qr(mul(A,  orth(randn(n, block_size)), 0), 0);
        % Ensuring orthogonalization of Q.
        Q_i = orth(Q_i - (Q * (Q' * Q_i)));
        % Deterministic computation of B_i.
        B_i = get_B(A, Q_i);
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
        fprintf("sqrt(||A||_F^2 - ||B||_F^2) / ||A||_F^2: %e\n", i, approximation_error);
        %fprintf("||A - QB||_F / ||A||_F %e\n", norm(A_cpy - Q*B, 'fro') / normest(A_cpy, 'fro'));
        %fprintf("||A - QB||_2 / ||A||_2 %e\n", norm(A_cpy - Q*B, 2) / normest(A_cpy, 2));
        fprintf("||B_i||_2 /||A||_2: %e\n", norm(B_i, 2) / norm2_A);
        %fprintf("||B_i||_2 / ||Delta_i-1||_2: %e\n", norm(B_i, 2) / normest(A, 2));

        if approximation_error < tol && ~indicator_fro
            fprintf("FRO TERMINATION CRITERIA REACHED AND COMPUTED AT RANK %d, iteration %d\n", size(Q, 2), i);
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

        %A = A - Q_i * B_i;
        A = update(A, Q_i, B_i);

    end
end

function [Q] = RSI(A, k, p)

    [m, n] = size(A{1});
    v = 2 * p + 1;

    % Odd number of passes over A.
    if(mod(v, 2) == 0)
        Omega = randn(m, k);
        
        if (v > 2)
            %[Q, ~] = lu(A' * Omega);
            [Q, ~] = lu(mul(A, Omega, 1));
        else
            [Q, ~] = qr(mul(A, Omega, 1), 0);
        end
    % Even number of passes over A.
    else
        Q = randn(n, k);
        if p == 0
            [Q, ~] = qr(mul(A, Q, 0), 0);
        end
    end

    for i = 1 : p
        [Q, ~] = lu(mul(A, Q, 0));
        [Q, ~] = lu(mul(A, Q, 1));
    end

    [Q, ~] = qr(mul(A, Q, 0), 0);
end