function Q_out = mul(A, Omega, transpose)
 % A consists of repeated [A, Q, B, Q, B, ...] and represents A -
 % sum(Q_i * B_i) forall i. 
 % To represent skecthing, we should multiply every odd element by
 % Omega and then actually compute Q
 numels = size(A);
 % Make sure we don't actually change any element of A
 if transpose
    % Q = A' * Omega
    Q_out = A{1}' * Omega;
    for i = 2:numels-1
        if mod(i, 2) == 0
             % Need to perform block multiplication to avoid instability
             b_sz = 100;
             M = [];
             start = 1;
             stop = b_sz;
             while stop <= size(Q_out , 1)
                    M = [M;  (A{i + 1}(:, start:stop))' * A{i}' * Omega]; %#ok<AGROW>
                    start = stop + 1;
                    stop = stop + b_sz;
             end
             Q_out = Q_out - M;

            %THIS EXPRESSION IS STABLE BUT REQUIRES STORAGE
            %Q_out = Q_out - (A{i + 1}' * A{i}' * Omega);
            %THIS EXPRESSION IS UNSTABLE, BUT SOLVES STORAGE ISSUE
            %Q_out = Q_out - (A{i + 1}'*(A{i}' * Omega));
        end
    end
 else
     % Q_out = A * Omega
     Q_out = A{1} * Omega;
     for i = 3:numels
         if mod(i, 2) ~= 0
             % Need to perform block multiplication to avoid instability
             b_sz = 100;
             M = [];
             start = 1;
             stop = b_sz;
             while stop <= size(Q_out , 1)
                    M = [M;  A{i - 1}(start:stop, :) * A{i} * Omega]; %#ok<AGROW>
                    start = stop + 1;
                    stop = stop + b_sz;
             end
             Q_out = Q_out - M;

             %THIS EXPRESSION IS STABLE BUT REQUIRES STORAGE
             %Q_out = Q_out - (A{i - 1}*A{i} * Omega);
             %THIS EXPRESSION IS UNSTABLE, BUT SOLVES STORAGE ISSUE
             %Q_out = Q_out - (A{i - 1}*(A{i} * Omega));
         end
     end
 end
end