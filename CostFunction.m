function y = CostFunction(X,Y,theta),
    % X is the design matrix containing our examples
    % y is the labels
    % theta is of the form [a,b]
    m = size(X,1);
    predictions = X*theta;
    sqrErrors = (predictions - Y) .^ 2;
    y = (1/(2*m))* sum(sqrErrors);
