function[] = QB_tests()

    %qb_rbki_blocked()
    qb_rbki()


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

function[] = qb_rbki_blocked()
    fprintf("/--------------------------------------------------------/\n")
    n = 100;
    tol = 1e-16;
    k = 50;
    block_size = 10;
    inner_block_size_factor = 2;
    num_iters = 2;

    A = zeros(1, n);
    A(:, 1:9) = [10, 9, 8, 7, 6, 5, 4, 3, 2];
    for i = 10:k
        A(1, i) = A(1, i-1) - (1 / n);
    end
    A = diag(A);
    %[A,~] = gen_exp_spectrum(n, n, k, 80);

    [err_ratio_vector1, err_vector] = QB_RBKI_deflation(A, block_size, inner_block_size_factor, tol, k, num_iters);
    
    figure();
    subplot(2, 1, 1);
    plot(err_ratio_vector1);
    ylabel('(err / err\_exact) - 1') 
    xlabel('Iteration') 
    %figure();
    subplot(2, 1, 2);
    plot(err_vector)
    ylabel('||A - QB|| / ||A||') 
    xlabel('Iteration')


    x=1:10;
%y1=x.^0.5;
%y2=x;
%y3=x.^2;
%y4=x.^3;
%figure1=figure('Position', [100, 100, 1024, 1200]);
%subplot(4,1,1);
%plot(x,y1);

    fprintf("/--------------------------------------------------------/\n")
end

function[] = qb_rbki()
    fprintf("/--------------------------------------------------------/\n")
    n = 100;
    tol = 1e-15;
    k = 50;
    block_size = 10;
    inner_block_size_factor = 2;
    num_iters = 10;

    A = zeros(1, n);
    A(:, 1:9) = [10, 9, 8, 7, 6, 5, 4, 3, 2];
    for i = 10:k
        A(1, i) = A(1, i-1) - (1 / n);
    end
    A = diag(A);
    %[A,~] = gen_exp_spectrum(n, n, k, 80);

    err_ratio_vector2 = [];
    err_vector = [];
    block_size_iter = block_size;
    for i = 1:(k / block_size)
        [err_ratio_entry, err_entry] = QB_RBKI(A, inner_block_size_factor, block_size_iter, num_iters);

        err_vector = [err_vector, err_entry];  %#ok<AGROW>
        err_ratio_vector2 = [err_ratio_vector2, err_ratio_entry];  %#ok<AGROW>
        block_size_iter = block_size_iter + block_size;
    end

    figure();
    subplot(2, 1, 1);
    plot(err_ratio_vector2);
    ylabel('(err / err\_exact) - 1') 
    xlabel('Iteration') 
    %figure();
    subplot(2, 1, 2);
    plot(err_vector)
    ylabel('||A - QB|| / ||A||') 
    xlabel('Iteration')



    fprintf("/--------------------------------------------------------/\n")
end