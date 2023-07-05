function[] = all_tests()
    rsi_blocked_slow_exp_lowrank()

    %qb_small_hard_case()
    %qb_small_hard_case_abstract_operator()
    %qb_large_hard_case_abstract_operator()
    %qb_exlarge_hard_case_abstract_operator()
    
    %qb_fast_exp_full_rank()
    %qb_slow_exp_full_rank()
    
    %qb_fast_exp_low_rank()
    %qb_slow_exp_low_rank()
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

%Runs a QB with abstarct operator on an n by n diagonal matrix A with 
% A(1, 1) = 10 and A(i, i) = 1 for i = 2...500.
% Expected outcome: fro termination criteria reached at rank 500, iteration
% 500 / b_sz; spec termination criteria reached ar rank 500, iteration (500
% / b_sz) + 1.
% This case techinally fits in my ram, but still am using it with an
% abstract operator for extra safety.

function[] = qb_small_hard_case_abstract_operator()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    A = zeros(1, n);
    A(1:500) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    A(1, 1) = 10;
    b_sz = 50;
    k = 600;
    tol = 1e-15;
    p = 2;
    A = {A};

    QB_blocked_pi_abstract_operator(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

% Same as above, but more info bc no abstarct operator
function[] = qb_small_hard_case()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    A = zeros(1, n);
    A(1:500) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    A(1, 1) = 10;
    b_sz = 50;
    k = 600;
    tol = 1e-15;
    p = 2;

    QB_blocked_pi(A, b_sz, tol, k, p, A);
    fprintf("/--------------------------------------------------------/\n")
end

% This does not fit in my memory anymore, takes ~2 hours to complete on my
% machine.
function[] = qb_large_hard_case_abstract_operator()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^5;
    A = zeros(1, n);
    A(1:500) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    A(1, 1) = 10;
    b_sz = 50;
    k = 600;
    tol = 1e-15;
    p = 2;
    A = {A};

    QB_blocked_pi_abstract_operator(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

% Extremely large case, suggested by Rob.
function[] = qb_exlarge_hard_case_abstract_operator()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^6;
    A = zeros(1, n);
    A(1:500) = 1;
    A = sparse(1:n, 1:n, A, n, n);
    A(1, 1) = 10;
    b_sz = 50;
    k = 600;
    tol = 1e-15;
    p = 2;
    A = {A};

    QB_blocked_pi_abstract_operator(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

% "Full-rank case" with very small singular values: may terminate early due
% to the size of last singular values
function[] = qb_fast_exp_full_rank()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    k = n;
    [A,~] = gen_exp_spectrum(n, n, k, 30);
    b_sz = 50;
    tol = 1e-15;
    p = 2;

    QB_blocked_pi(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

% Full-rank case: we expect spec metric to not work, since it is supposed
% to detect rank at iteration i+1 (which will be off-limits in a full-rank
% case);
function[] = qb_slow_exp_full_rank()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    k = n;
    [A,~] = gen_exp_spectrum(n, n, k, 80);
    b_sz = 50;
    tol = 1e-15;
    p = 2;

    QB_blocked_pi(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

function[] = qb_fast_exp_low_rank()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    k = 600;
    [A,~] = gen_exp_spectrum(n, n, 500, 30);
    b_sz = 50;
    tol = 1e-15;
    p = 2;

    QB_blocked_pi(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

function[] = qb_slow_exp_low_rank()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    k = 600;
    [A,~] = gen_exp_spectrum(n, n, 500, 80);
    b_sz = 50;
    tol = 1e-15;
    p = 2;

    QB_blocked_pi(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

function[] = rsi_blocked_slow_exp_lowrank()
    fprintf("/--------------------------------------------------------/\n")
    n = 10^3;
    k = 600;
    [A,~] = gen_exp_spectrum(n, n, 500, 80);
    b_sz = 50;
    tol = 1e-14;
    p = 2;

    RSI_blocked(A, b_sz, tol, k, p);
    fprintf("/--------------------------------------------------------/\n")
end

