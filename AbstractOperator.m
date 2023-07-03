classdef AbstractOperator
   properties
      Value {mustBeNumeric}
   end
   methods
       
      function A = update(A, Q, B)
         % A is a cell array
         % Data is a cell array that takes Q, B into account
         A = [A, Q, B];
      end

      function Q_out = mul(A, Omega, transpose)
         % A consists of repeated [A, Q, B, Q, B, ...] and represents A -
         % sum(Q_i * B_i) forall i. 
         % To represent skecthing, we should multiply every odd element by
         % Omega and then actually compute Q
         numels = size(A);
         % Make sure we don't actually change any element of A
         A_cpy = A;
         if transpose
            % Q = A' * Omega
            Q_out = A_cpy(1)' * Omega;
            for i = 2:numels-1
                % transpose every element
                A_cpy(i) = A_cpy(i)';
                if mod(i, 2) == 0
                    % Q' = Q' * Omega
                    A_cpy(i) = A_cpy(i) * Omega;
                    %Q_out = A' * Omega - B'Q' * Omega
                    Q_out = Q_out - A_cpy(i + 1) * A_cpy(i);
                end
            end
         else
             % Q_out = A * Omega
             Q_out = A_cpy(1) * Omega;
             for i = 3:numels
                 if mod(i, 2) ~= 0
                     % B = B * Omega
                     A_cpy(i) = A_cpy(i) * Omega;
                     % Q_out = A * Omega - QB * Omega
                     Q_out = Q_out - (A_cpy(i - 1) * A_cpy(i));
                 end
             end
         end
      end

      function B_out = get_B(A, Q_in)
         numels = size(A);
         % Make sure we don't actually change any element of A
         A_cpy = A;
         
         B_out = Q_in * A;

         for i = 2:numels - 1
           if mod(i, 2) == 0
                % Q = Q_in' * Q
                A_cpy(i) = Q_in' * A_cpy(i);
                %Q_out = A' * Omega - B'Q' * Omega
                B_out = B_out - A_cpy(i) * A_cpy(i + 1);
            end
         end
      end
   end
end




