function[] = qb_testing_proper_randlapack()

    %m = 10000;
    %A = randn(m, m/2);
    %k = 5000;
    %p = 7;
    %b_sz = 50;
    %tol = 1e-12;
    %A_cpy = A;

    A = zeros(1, 1000);
    A(1:500) = 0.00001;
    A(501:1000) = 0.00001;
    A(1) = 1;
    A = diag(A);
    A_cpy = A;
    b_sz = 50;
    k = 1000;
    tol = 1e-15;
    p = 1;

    [Q,B] = qb_2(A, b_sz, tol, k, p, 0, A_cpy);
    fprintf("SIZE OF Q IS %d by %d\n", size(Q, 1), size(Q, 2));
    fprintf("QUALITY OF B ERROR IS %e\n", norm(Q' * A_cpy - B, 'fro') / norm(B,'fro'));
end

function [t] = MarlaRandStream(s)
    if isa(s, 'RandStream')
        t = s;
    else
        seed = mod(s, 2^32);
        t = RandStream('mt19937ar', 'Seed', seed, 'NormalTransform', 'Ziggurat');
    end
end
function [A, s] = gen_exp_spectrum(m, n, k, t, s)
    s = MarlaRandStream(s);
    spectrum = exp((1 : k) / -t);
    [A, s] = gen_test_mat(m, n, k, spectrum, s);
end
function [A, S] = gen_test_mat(m, n, k, spectrum, s)
    s = MarlaRandStream(s);
    Buf = randn(s, m, k);
    [U, ~] = qr(Buf, 0);
    Buf = randn(s, n, k);
    [V, ~] = qr(Buf, 0);
    if isscalar(spectrum)
        spectrum = abs(randn(s, 1, k));
        spectrum = sort(spectrum,'descend');
    end
    S = spdiags(spectrum', 0, k, k);
    A = U * S * V';
end
function [Q, B] = qb_2(A, block_size, tol, k, p, s, A_cpy)
    s = MarlaRandStream(s);
    norm_A = norm(A, 'fro');
    norm2_A = norm(A, 2);
    % Early termination check on an empty input.
    if norm_A == 0
        fprintf('The input matrix is empty.');
        return
    end
    % Setting initial error to zero.
    approximation_error = 0;
    class_A = class(A);
    [m, n] = size(A);
    norm_B = 0;
    % Pre-initialization of output matrices.
    Q = zeros(m, 0, class_A);
    B = zeros(0, n, class_A);
    % Iterative stage.
    for i = 1 : ceil(k / block_size)
        %fprintf(“K in QB %d\n”, k);
        if size(B, 1) + block_size > k
            block_size = k - size(B, 1);
        end
        % Consstructiong a sketch for current iteration.
        %Q_i = rf1_simplified(A, block_size, p);
        %Q_i = rf1(A, block_size, p);
        %Q_i = A * randn(n, block_size);
        Q_i = RSI(A, block_size, p, s);
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
        size(B)
        % Condition of reaching tolerance.
        %fprintf("COND(Q) AT ITERATION %d IS %e\n", i, cond(Q));
        %fprintf("COND(Q_i) AT ITERATION %d IS %e\n", i, cond(Q_i));
        %fprintf("NORM_A - NORM_B %e\n", 1 - norm_B/norm_A);
        fprintf("APPRIXIMATION ERROR COMPUTED FAST AT ITERATION %d IS %e\n", i, approximation_error);
        slow_error = norm(A_cpy - Q*B, 'fro') / norm(A_cpy, 'fro');
        fprintf("APPRIXIMATION ERROR COMPUTED SLOW AT ITERATION %d IS %e\n", i, slow_error);
        fprintf("||B_i||_2 <= ||Q_i'Delta_i||_2  /||A||_2: %e\n", norm(B_i, 2)/norm2_A);

        % NEW CRITERION
        if(1 - norm_B/norm_A <= 2 * eps('double'))
            fprintf("NEW TERMINATION CRITERIA REACHED\n");
            %break;
        end


        if approximation_error < tol
            fprintf("TERMINATION CRITERIA REACHED\n");
            break;
        end
        A = A - Q_i * B_i;
        %break;
    end
end
function [Q] = rf1(A, k, p)
    [K] = rs_krylov(A, k, p);
    [Q, ~] = qr(K, 0);
end
function [K] = rs_krylov(A, k, p)

    [m, n] = size(A);
    fprintf("\n block_size is %d\n", k);
    K = [];
    if(mod(p, 2) == 0)
        % Adding x to K does not cound as a pass
        numcols = ceil(k / ((p / 2) + 1));
        [x, ~] = qr(randn(m, numcols), 0);
        K = x;

        fprintf("SIZE OF A KRYLOV BLOCK IS %d\n", numcols);
    else % odd iterations
        % Adding Ax to K counts as a pass
        numcols = ceil(k / ((p - 1) / 2));
        [x, ~] = qr(A * randn(n, numcols), 0);
        K = x;
        p = p - 1;
    end
    for i = 1:(p/2)
        % Riley's method
        %AtK = orth(A' * K);
        %K = [K, A*AtK];

        % My method
        [x, ~] = qr(A' * x, 0);
        
        x = A * x;
        K = [K, x];
    end
    % safety measure
    if(size(K, 2) > k)
            K = K(:, 1:k);
    end
end

function [Q] = RSI(A, k, p, s)

    class_A = class(A);
    [m, n] = size(A);
    v = 2 * p + 1;

    % Odd number of passes over A.
    if(mod(v, 2) == 0)
        % By default, a Gaussian random sketching matrix is used.
        % Alternative choices are present in '../Sketching_Operators'
        Omega = randn(s, m, k, class_A);
        
        if (v > 2)
            [Q, ~] = lu(A' * Omega);
        else
            [Q, ~] = qr(A' * Omega, 0);
        end
    % Even number of passes over A.
    else
        % By default, a Gaussian random sketching matrix is used.
        % Alternative choices are present in
        % '../../utils/sketching_operators'.
        Q = randn(s, n, k, class_A);
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