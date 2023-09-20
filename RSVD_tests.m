function[] = RSVD_tests()
    Rob_test(5 * 10^4);
    Rob_test(5 * 10^5);
    Rob_test(5 * 10^6);
    %RBKI_incremental_test();
end

% Generator for random matrix with an exponentially decaying spectrum
function [A, s] = gen_exp_spectrum(m, n, k, t)
    spectrum = exp((1 : k) / -t);
    [A, s] = gen_test_mat(m, n, k, spectrum);
end
function [A, S] = gen_test_mat(m, n, k, spectrum)
    Buf = randn(m, k);
    [U, ~] = qr(Buf, 0);
    Buf = randn(n, k);
    [V, ~] = qr(Buf, 0);
    if isscalar(spectrum)
        spectrum = abs(randn(1, k));
        spectrum = sort(spectrum,'descend');
    end
    S = spdiags(spectrum', 0, k, k);
    A = U * S * V';
end

function[] = RBKI_incremental_test()
    fprintf("/--------------------------------------------------------/\n")
    n = 1000;
    tol = 1e-16;
    k = n;
    inner_block_size_factor = 2;

    %A = zeros(1, n);
    %A(:, 1:9) = [10, 9, 8, 7, 6, 5, 4, 3, 2];
    %for i = 10:k
    %    A(1, i) = A(1, i-1) - (1 / n);
    %end
    %A = diag(A);
    %A_cpy = A;
    [A,~] = gen_exp_spectrum(n, n, n, 80);
    A_cpy = A;

    [U, Sigma, V] = RBKI_incremental_final(A, inner_block_size_factor * k, tol);

    norm(A_cpy - U * diag(Sigma) * V', 'fro') / norm(A_cpy, 'fro')

    fprintf("/--------------------------------------------------------/\n")
end

function[] = Rob_test(n)
    fprintf("/--------------------------------------------------------/\n")
    tol = sqrt(eps('double'));

    %n = 5 * 10^4;
    A = zeros(1, n);
    for i = 1:n
        A(1, i) = max(exp(-i/25), (1 - i/10^5)/25);
    end
    A = sparse(1:n, 1:n, A, n, n);
    A_1 = A;
    A_2 = A;

    tic;
    [U, Sigma, V] = RBKI_incremental_final(A, n, tol);
    fprintf("RBKI takes %d seconds\n", toc);
    norm(A_2 - U * diag(Sigma) * V', 'fro') / norm(A_2, 'fro')

    tic;
    [U, Sigma, V] = svdsketch(A_1, tol);
    fprintf("RSVD takes %d seconds\n", toc);
    norm(A_2 - U * Sigma * V', 'fro') / norm(A_2, 'fro')

    fprintf("/--------------------------------------------------------/\n")
end
