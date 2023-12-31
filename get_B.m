  function B_out = get_B(A, Q_in)
     numels = size(A);
     % Make sure we don't actually change any element of A
     
     B_out = Q_in' * A{1};
     for i = 2:numels - 1
       if mod(i, 2) == 0
            %Q_out = Q_in' * A - Q_in' * QB
            B_out = B_out - (Q_in' * A{i} * A{i + 1});
        end
     end
  end