      function A = update(A, Q, B)
         % A is a cell array
         % Data is a cell array that takes Q, B into account
         A = [A; Q];
         A = [A; B];
      end